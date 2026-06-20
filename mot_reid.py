import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Add the local directory to sys.path so we can import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Add typical ByteTrack paths to sys.path
for path in ['./ByteTrack', '../ByteTrack', '/content/ByteTrack', os.path.join(current_dir, 'ByteTrack')]:
    if os.path.exists(path):
        sys.path.append(os.path.abspath(path))

# Check for yolox import to help the user diagnose setup issues
try:
    from yolox.tracker.byte_tracker import BYTETracker
except ImportError:
    print("WARNING: 'yolox' module not found. Please ensure ByteTrack is cloned and setup.py is run.")
    print("Refer to installation details in README.md or run 'python notebooks/stages/stage_01.py' in Colab.")

from utils.tracker import get_tracker, bbox_iou
from utils.reid import build_reid_model, crop_and_preprocess, extract_features, aggregate_embeddings, match_embedding
from utils.visualization import annotate_frame, concatenate_side_by_side

def process_frame(frame, yolo_model, reid_model, byte_tracker, track_embeddings, gallery, class_ids, match_thresh, device):
    """
    Core pipeline: Detect, Track, Extract Features, Match Embeddings, Annotate.
    """
    results = yolo_model(frame)[0]
    detections = results.boxes.cpu().numpy()
    xyxy = detections.xyxy
    conf = detections.conf
    cls = detections.cls

    # Filter detections by target class IDs
    mask = np.isin(cls, class_ids)
    xyxy = xyxy[mask]
    conf = conf[mask]
    cls = cls[mask]

    if len(xyxy) == 0:
        return frame

    # Prepare detections for BYTETracker (x1, y1, x2, y2, score)
    dets = np.hstack((xyxy, conf[:, None]))
    tracks = byte_tracker.update(dets, frame.shape, frame.shape)

    # Associate detections with track IDs
    tids = []
    for det in xyxy:
        tid = None
        for trk in tracks:
            if bbox_iou(det, trk.tlbr) > 0.5:
                tid = trk.track_id
                break
        tids.append(tid)
    tids = np.array(tids)

    # Collect valid crops for ReID feature extraction
    crops = []
    tids_valid = []
    for bbox, tid in zip(xyxy, tids):
        if tid is None:
            continue
        crop = crop_and_preprocess(frame, bbox, device=device)
        if crop is not None:
            crops.append(crop)
            tids_valid.append(tid)

    # Extract appearance features
    feats = extract_features(crops, reid_model) if crops else None
    if feats is not None:
        for tid, feat in zip(tids_valid, feats):
            if tid not in track_embeddings:
                track_embeddings[tid] = []
            track_embeddings[tid].append(feat)

    # Assign/recover global IDs using appearance matching against the gallery
    global_ids = {}
    for tid, embs in track_embeddings.items():
        agg_emb = aggregate_embeddings(embs)
        if agg_emb is None:
            continue
        
        # Match against gallery
        matched_entry, similarity = match_embedding(agg_emb, gallery, threshold=match_thresh)
        if matched_entry is not None:
            global_ids[tid] = matched_entry['global_id']
        else:
            # Create new identity in the gallery
            new_gid = len(gallery) + 1
            gallery.append({'global_id': new_gid, 'embedding': agg_emb})
            global_ids[tid] = new_gid

    # Generate annotation labels
    labels = []
    for tid, c, conf_score in zip(tids, cls, conf):
        if tid is None:
            labels.append("")
        else:
            gid = global_ids.get(tid, -1)
            class_name = yolo_model.model.names[int(c)]
            labels.append(f"GID:{gid} TID:{tid} {class_name} {conf_score:.2f}")

    # Annotate frame with bounding boxes and labels
    frame = annotate_frame(frame, xyxy, labels)
    return frame

def run_pipeline(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")

    # Load YOLO model
    print(f"Loading YOLO model from: {args.yolo_weights}")
    yolo_model = YOLO(args.yolo_weights)
    yolo_model.fuse()

    # Build ReID model
    print("Building ReID model (OSNet)...")
    reid_model = build_reid_model(
        name=args.reid_model_name,
        weights_path=args.reid_weights,
        device=device,
        pretrained=(args.reid_weights is None)
    )

    # Determine execution mode (Single or Dual camera)
    is_dual = args.source2 is not None
    
    if is_dual:
        print(f"Running in Dual-Camera mode:")
        print(f" - Source 1: {args.source}")
        print(f" - Source 2: {args.source2}")
        cap1 = cv2.VideoCapture(args.source)
        cap2 = cv2.VideoCapture(args.source2)
        
        if not cap1.isOpened() or not cap2.isOpened():
            print("Error: Could not open one or both video sources.")
            return

        fps1 = int(cap1.get(cv2.CAP_PROP_FPS)) or 30
        fps2 = int(cap2.get(cv2.CAP_PROP_FPS)) or 30
        
        byte_tracker_1 = get_tracker(frame_rate=fps1)
        byte_tracker_2 = get_tracker(frame_rate=fps2)
        
        track_embeddings_1 = {}
        track_embeddings_2 = {}
        gallery = []  # Shared across both cameras
        
        # Prepare output writer
        writer = None
        if args.output:
            width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)) + int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = max(int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
            writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), min(fps1, fps2), (width, height))
            print(f"Saving output video to: {args.output}")

        frame_count = 0
        start_time = time.time()
        try:
            while True:
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                if not ret1 or not ret2:
                    break

                # Process individual camera streams
                frame1 = process_frame(
                    frame1, yolo_model, reid_model, byte_tracker_1, 
                    track_embeddings_1, gallery, args.classes, args.match_thresh, device
                )
                frame2 = process_frame(
                    frame2, yolo_model, reid_model, byte_tracker_2, 
                    track_embeddings_2, gallery, args.classes, args.match_thresh, device
                )

                # Concatenate side-by-side
                combined = concatenate_side_by_side(frame1, frame2)

                if writer:
                    writer.write(combined)

                if args.show:
                    cv2.imshow("Dual Camera MOT-ReID", combined)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_count += 1
                if frame_count % 10 == 0:
                    fps_est = frame_count / (time.time() - start_time)
                    print(f"Processed {frame_count} frames | Est. FPS: {fps_est:.2f} | Gallery Size: {len(gallery)}")

        except KeyboardInterrupt:
            print("Processing interrupted by user.")
        finally:
            cap1.release()
            cap2.release()
            if writer:
                writer.release()
            if args.show:
                cv2.destroyAllWindows()
            print("Released resources.")
            
    else:
        print(f"Running in Single-Camera mode:")
        print(f" - Source: {args.source}")
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {args.source}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        byte_tracker = get_tracker(frame_rate=fps)
        track_embeddings = {}
        gallery = []

        writer = None
        if args.output:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
            writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            print(f"Saving output video to: {args.output}")

        frame_count = 0
        start_time = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = process_frame(
                    frame, yolo_model, reid_model, byte_tracker, 
                    track_embeddings, gallery, args.classes, args.match_thresh, device
                )

                if writer:
                    writer.write(frame)

                if args.show:
                    cv2.imshow("MOT-ReID", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_count += 1
                if frame_count % 10 == 0:
                    fps_est = frame_count / (time.time() - start_time)
                    print(f"Processed {frame_count} frames | Est. FPS: {fps_est:.2f} | Gallery Size: {len(gallery)}")

        except KeyboardInterrupt:
            print("Processing interrupted by user.")
        finally:
            cap.release()
            if writer:
                writer.release()
            if args.show:
                cv2.destroyAllWindows()
            print("Released resources.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Object Tracking with Re-Identification Pipeline")
    parser.add_argument("--source", type=str, required=True, help="Path to primary input video or camera index")
    parser.add_argument("--source2", type=str, default=None, help="Path to secondary input video (for dual-camera ReID)")
    parser.add_argument("--yolo-weights", "--weights", type=str, default="yolov8n.pt", help="Path to YOLO weights (.pt)")
    parser.add_argument("--reid-model-name", type=str, default="osnet_x1_0", help="TorchReID model name")
    parser.add_argument("--reid-weights", type=str, default=None, help="Path to pre-trained ReID checkpoint (.pth)")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated output video")
    parser.add_argument("--match-thresh", type=str, default=0.7, help="Cosine similarity threshold for ReID matching")
    parser.add_argument("--classes", type=int, nargs="+", default=[2, 3, 5, 7], help="List of COCO class IDs to detect/track (default: vehicle classes)")
    parser.add_argument("--show", action="store_true", help="Display tracking output in window (using cv2.imshow)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")

    args = parser.parse_args()
    
    # Standardize types
    args.match_thresh = float(args.match_thresh)
    
    # Try converting source to int if it represents a camera index
    if args.source.isdigit():
        args.source = int(args.source)
    if args.source2 and args.source2.isdigit():
        args.source2 = int(args.source2)

    run_pipeline(args)
