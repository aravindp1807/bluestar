# Multi-Object Tracking with Re-Identification (MOT-ReID)

A robust pipeline for **multi-object tracking (MOT)** combined with **person re-identification (ReID)** to maintain consistent identities across occlusions, camera cuts, and long-term tracking scenarios. Built on **YOLOv8** for detection, **ByteTrack** for association, and **TorchReID** for appearance-based identity resolution.

---

## 📋 Overview

This project implements a complete **tracking-by-detection** framework enhanced with a **ReID gallery** to recover identities after track fragmentation. The pipeline processes video streams frame-by-frame:

1. **Detection** – YOLOv8 predicts bounding boxes for target classes (default: `person`).
2. **Tracking** – ByteTrack associates detections into tracklets using Kalman filtering and IoU matching, handling both high- and low-confidence detections.
3. **Feature Extraction** – Each tracklet crop is passed through a TorchReID backbone (OSNet, ResNet-50, etc.) to compute a 512-D embedding.
4. **Embedding Aggregation** – Per-track embeddings are averaged (or max-pooled) to form a stable track signature.
5. **Gallery Matching** – New track embeddings are compared against a persistent identity gallery via cosine similarity; matches above threshold reuse existing IDs, otherwise a new identity is registered.
6. **Annotation & Output** – Annotated frames are written to disk with ID labels, bounding boxes, and optional trajectory trails.

The notebook is designed to run end-to-end in **Google Colab** (GPU-enabled) but can be adapted for local execution.

---

## ✨ Features

| Component | Library / Implementation | Purpose |
|-----------|--------------------------|---------|
| **Object Detection** | `ultralytics.YOLO` (YOLOv8n–x) | Fast, accurate person detection |
| **Multi-Object Tracking** | `yolox.tracker.byte_tracker.BYTETracker` | Association with low-score recovery |
| **Re-Identification** | `torchreid` (OSNet, ResNet, etc.) | Discriminative appearance embeddings |
| **Video I/O & Annotation** | `cv2`, `supervision` | Frame reading, drawing, video writing |
| **Embedding Ops** | `torch`, `numpy`, `torch.nn.functional` | L2-normalization, cosine similarity, aggregation |
| **Environment** | Google Colab / local Python ≥3.8 | GPU-accelerated inference |

**Key capabilities**
- 🔄 **Identity persistence** across occlusions and track breaks
- 🎯 **Configurable similarity threshold** (`match_embedding(threshold=0.7)`)
- 📦 **Modular functions** for easy integration into other pipelines
- 📹 **Output video** with annotated IDs, boxes, and optional trails
- ⚡ **Batch feature extraction** for throughput optimization

---

## 🛠 Installation

### 1. Clone & Install Dependencies (Colab-ready)

```bash
# Core ML & tracking libraries
pip install ultralytics torchreid supervision==0.1.0

# ByteTrack (official repo with Cython IoU)
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
sed -i 's/onnx==1.8.1/onnx==1.9.0/g' requirements.txt
pip install -r requirements.txt
pip install cython_bbox onemetric loguru lap thop
python3 setup.py develop
cd ..

# Ensure ByteTrack is on sys.path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/ByteTrack"
```

### 2. Local Environment (Optional)

```bash
conda create