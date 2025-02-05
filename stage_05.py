# Stage 5: feat: configure device detection and CUDA setup
# ==================================================

# === Step 1: Upload videos from your local machine ===
from google.colab import files

print("Upload Cam 1 video:")
uploaded_cam1 = files.upload()
VIDEO_PATH_1 = next(iter(uploaded_cam1))

print("Upload Cam 2 video:")
uploaded_cam2 = files.upload()
VIDEO_PATH_2 = next(iter(uploaded_cam2))

print(f"Cam 1 video file: {VIDEO_PATH_1}")
print(f"Cam 2 video file: {VIDEO_PATH_2}")

# === Step 2: Imports and device setup ===
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from types import SimpleNamespace
import torchreid
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Step 3: Define helper functions (reuse from your existing code) ===
def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)
    return iou

def annotate_frame(frame, boxes, labels):
    for bbox, label in zip(boxes, labels):
        if not label:
            continue
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def crop_and_preprocess(frame, bbox):
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return preprocess(crop).unsqueeze(0).to(device)

@torch.no_grad()
def extract_features(crops, model):
    if len(crops) == 0:
        return None
    batch = torch.cat(crops, dim=0)
    feats = model(batch)
    feats = F.normalize(feats, dim=1)
    return feats.cpu().numpy()

def aggregate_embeddings(embs):
    if embs is None or len(embs) == 0:
        return None
    mean_emb = np.mean(embs, axis=0)
    return mean_emb / (np.linalg.norm(mean_emb) + 1e-6)

def match_embedding(query_emb, gallery, threshold=0.7):
    if len(gallery) == 0:
        return None, 0.0
    gallery_feats = np.stack([entry['embedding'] for entry in gallery])
    sims = np.dot(query_emb, gallery_feats.T)
    max_idx = np.argmax(sims)
    if sims[max_idx] >= threshold:
        return gallery[max_idx], sims[max_idx]
    return None, sims[max_idx]

# === Step 4: Load models ===

YOLO_MODEL_PATH = "/content/yolo11n (1).pt"  # Update if your YOLO weight file name is different
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_model.fuse()

args = SimpleNamespace(
    track_thresh=0.3,
    track_buffer=30,
    match_thresh=0.8,
    aspect_ratio_thresh=3.0,
    min_box_area=10,
    mot20=False
)

byte_tracker_1 = BYTETracker(args, frame_rate=30)
byte_tracker_2 = BYTETracker(args, frame_rate=30)

osnet = torchreid.models.build_model(name='osnet_x1_0', num_classes=0, pretrained=False)
osnet.to(device)
osnet.eval()

osnet_weights_path = "/content/osnetpretrained.pth"  # Update if your OSNet weight file name is different
checkpoint = torch.load(osnet_weights_path, map_location=device)
state_dict = checkpoint.get('state_dict', checkpoint)
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier.')}
osnet.load_state_dict(filtered_state_dict, strict=False)

CLASS_ID = [2,3,5,7]  # car, motorcycle, bus, truck

# === Step 5: Setup tracking and embedding containers ===
gallery = []

track_embeddings_1 = {}
track_embeddings_2 = {}

# === Step 6: Frame processing function ===
def process_frame(frame, byte_tracker, track_embeddings, gallery):
    results = yolo_model(frame)[0]
    detections = results.boxes.cpu().numpy()
    xyxy = detections.xyxy
    conf = detections.conf
    cls = detections.cls

    mask = np.isin(cls, CLASS_ID)
    xyxy = xyxy[mask]
    conf = conf[mask]
    cls = cls[mask]

    dets = np.hstack((xyxy, conf[:, None]))
    tracks = byte_tracker.update(dets, frame.shape, frame.shape)

    tids = []
    for det in xyxy:
        tid = None
        for trk in tracks:
            if bbox_iou(det, trk.tlbr) > 0.5:
                tid = trk.track_id
                break
        tids.append(tid)
    tids = np.array(tids)

    crops = []
    tids_valid = []
    for bbox, tid in zip(xyxy, tids):
        if tid is None:
            continue
        crop = crop_and_preprocess(frame, bbox)
        if crop is not None:
            crops.append(crop)
            tids_valid.append(tid)
    feats = extract_features(crops, osnet) if crops else None
    if feats is not None:
        for tid, feat in zip(tids_valid, feats):
            if tid not in track_embeddings:
                track_embeddings[tid] = []
            track_embeddings[tid].append(feat)

    # Assign global IDs
    global_ids = {}
    for tid, embs in track_embeddings.items():
        agg_emb = aggregate_embeddings(embs)
        if agg_emb is None:
            continue
        max_sim = 0
        matched_gid = None
        for entry in gallery:
            sim = np.dot(agg_emb, entry['embedding'])
            if sim > max_sim:
                max_sim = sim
                matched_gid = entry['global_id']
        if max_sim > 0.7:
            global_ids[tid] = matched_gid
        else:
            new_gid = len(gallery) + 1
            gallery.append({'global_id': new_gid, 'embedding': agg_emb})
            global_ids[tid] = new_gid

    labels = []
    for tid, c, conf_score in zip(tids, cls, conf):
        if tid is None:
            labels.append("")
        else:
            gid = global_ids.get(tid, -1)
            labels.append(f"GID:{gid} TID:{tid} {yolo_model.model.names[int(c)]} {conf_score:.2f}")

    frame = annotate_frame(frame, xyxy, labels)
    return frame

# === Step 7: Process and visualize side-by-side ===

cap1 = cv2.VideoCapture(VIDEO_PATH_1)
cap2 = cv2.VideoCapture(VIDEO_PATH_2)

print("Starting side-by-side visualization in Colab. Press interrupt (Ctrl+C) to stop.")

try:
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        frame1 = process_frame(frame1, byte_tracker_1, track_embeddings_1, gallery)
        frame2 = process_frame(frame2, byte_tracker_2, track_embeddings_2, gallery)

        # Resize frames for consistent height to concat horizontally
        height = max(frame1.shape[0], frame2.shape[0])
        if frame1.shape[0] != height:
            ratio = height / frame1.shape[0]
            frame1 = cv2.resize(frame1, (int(frame1.shape[1]*ratio), height))
        if frame2.shape[0] != height:
            ratio = height / frame2.shape[0]
            frame2 = cv2.resize(frame2, (int(frame2.shape[1]*ratio), height))

        combined = np.hstack((frame1, frame2))
        combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

        clear_output(wait=True)
        plt.figure(figsize=(20,10))
        plt.axis('off')
        plt.imshow(combined)
        display(plt.gcf())
        time.sleep(0.03)  # ~30 FPS

except KeyboardInterrupt:
    print("Visualization stopped.")

cap1.release()
cap2.release()
plt.close()