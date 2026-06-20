import numpy as np
from types import SimpleNamespace
from yolox.tracker.byte_tracker import BYTETracker

def bbox_iou(boxA, boxB):
    """
    Computes IoU between two bounding boxes in [x1, y1, x2, y2] format.
    """
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

def get_tracker(track_thresh=0.3, track_buffer=30, match_thresh=0.8, aspect_ratio_thresh=3.0, min_box_area=10, mot20=False, frame_rate=30):
    """
    Creates and returns an instance of BYTETracker.
    """
    args = SimpleNamespace(
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        aspect_ratio_thresh=aspect_ratio_thresh,
        min_box_area=min_box_area,
        mot20=mot20
    )
    return BYTETracker(args, frame_rate=frame_rate)
