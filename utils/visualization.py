import cv2
import numpy as np

def annotate_frame(frame, boxes, labels, color=(0, 255, 0), thickness=2):
    """
    Draws bounding boxes and labels onto the frame.
    """
    for bbox, label in zip(boxes, labels):
        if not label:
            continue
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            frame, 
            label, 
            (x1, max(0, y1 - 10)), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            color, 
            thickness
        )
    return frame

def concatenate_side_by_side(frame1, frame2):
    """
    Resizes two frames to have matching heights and concatenates them horizontally.
    """
    height = max(frame1.shape[0], frame2.shape[0])
    
    if frame1.shape[0] != height:
        ratio = height / frame1.shape[0]
        frame1 = cv2.resize(frame1, (int(frame1.shape[1] * ratio), height))
        
    if frame2.shape[0] != height:
        ratio = height / frame2.shape[0]
        frame2 = cv2.resize(frame2, (int(frame2.shape[1] * ratio), height))
        
    combined = np.hstack((frame1, frame2))
    return combined
