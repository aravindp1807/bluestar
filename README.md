# 🚀 Multi-Object Tracking with Re-Identification (MOT-ReID)

> A robust AI-powered surveillance and tracking pipeline that combines object detection, multi-object tracking, and person re-identification to maintain consistent identities across occlusions, camera transitions, and long-term tracking scenarios.

---

## 📖 Overview

Multi-Object Tracking (MOT) systems often struggle when objects disappear temporarily due to:

* Occlusions
* Crowded environments
* Camera motion
* Detection failures
* Re-entry into the scene

This project enhances traditional tracking by integrating **Person Re-Identification (ReID)**, enabling the system to recover and maintain identities even after tracks are lost.

The pipeline combines:

* **YOLOv8** for object detection
* **ByteTrack** for online tracking
* **OSNet / TorchReID** for appearance-based feature extraction
* **Cosine Similarity Matching** for identity recovery

The result is a scalable MOT-ReID framework capable of tracking individuals across long video sequences while preserving consistent IDs.

---

# 🎯 Key Features

### 🔍 Real-Time Person Detection

Utilizes YOLOv8 for accurate and fast person detection.

* YOLOv8n
* YOLOv8s
* YOLOv8m
* YOLOv8l
* YOLOv8x

supported interchangeably.

---

### 🎥 Multi-Object Tracking

ByteTrack performs robust association using:

* Kalman Filtering
* IoU Matching
* Low-confidence Recovery

allowing smoother trajectory generation and reduced ID switching.

---

### 🧠 Person Re-Identification

TorchReID extracts appearance embeddings using state-of-the-art backbones:

* OSNet
* ResNet50
* MobileNetV2
* DenseNet

Each person is represented by a discriminative feature vector.

---

### 🔄 Identity Recovery

When a track disappears:

1. Embeddings are stored in a gallery.
2. New tracks are compared against stored identities.
3. Cosine similarity determines matches.
4. Existing IDs are reassigned if similarity exceeds threshold.

This dramatically reduces identity fragmentation.

---

### 📹 Annotated Video Generation

Outputs include:

* Bounding Boxes
* Track IDs
* Confidence Scores
* Identity Labels
* Motion Trails

rendered directly into output videos.

---

## 🏗 System Architecture

```text
Input Video
      │
      ▼
YOLOv8 Detection
      │
      ▼
ByteTrack Association
      │
      ▼
Track Cropping
      │
      ▼
TorchReID Feature Extraction
      │
      ▼
Embedding Aggregation
      │
      ▼
Identity Gallery
      │
      ▼
Cosine Similarity Matching
      │
      ▼
Identity Assignment
      │
      ▼
Annotated Output Video
```

---

# ⚙️ Technology Stack

| Component           | Technology   |
| ------------------- | ------------ |
| Detection           | YOLOv8       |
| Tracking            | ByteTrack    |
| Re-Identification   | TorchReID    |
| Feature Backbone    | OSNet        |
| Video Processing    | OpenCV       |
| Annotation          | Supervision  |
| Deep Learning       | PyTorch      |
| Numerical Computing | NumPy        |
| Deployment          | Google Colab |

---

# 📦 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/MOT-ReID.git
cd MOT-ReID
```

---

## Install Dependencies

```bash
pip install ultralytics
pip install torchreid
pip install supervision==0.1.0
pip install opencv-python
pip install numpy
```

---

## Install ByteTrack

```bash
git clone https://github.com/ifzhang/ByteTrack.git

cd ByteTrack

sed -i 's/onnx==1.8.1/onnx==1.9.0/g' requirements.txt

pip install -r requirements.txt

pip install cython_bbox
pip install onemetric
pip install loguru
pip install lap
pip install thop

python setup.py develop

cd ..
```

---

## Configure ByteTrack Path

```python
import sys

sys.path.append("/content/ByteTrack")
```

---

# 🚀 Usage

## Load YOLO Model

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
```

---

## Initialize ReID Model

```python
import torchreid

reid_model = torchreid.models.build_model(
    name="osnet_x1_0",
    num_classes=1000,
    pretrained=True
)
```

---

## Run Tracking Pipeline

### Single-Camera Mode:
```bash
python mot_reid.py \
    --source input.mp4 \
    --weights yolov8n.pt \
    --output results.mp4
```

### Dual-Camera Mode (Cross-Camera Re-ID):
```bash
python mot_reid.py \
    --source input_cam1.mp4 \
    --source2 input_cam2.mp4 \
    --weights yolov8n.pt \
    --output results_combined.mp4
```

---

# 📂 Project Structure

```text
MOT-ReID/
│
├── data/
│   ├── input_videos/
│   └── outputs/
│
├── models/
│   ├── yolo/
│   └── reid/
│
├── notebooks/
│   ├── final_parking_osnet_bytetrack_yolo.ipynb
│   └── stages/
│       └── stage_01.py ... stage_36.py
│
├── utils/
│   ├── __init__.py
│   ├── tracker.py
│   ├── reid.py
│   └── visualization.py
│
├── mot_reid.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🔬 Re-Identification Workflow

### Step 1: Detection

YOLO detects persons in every frame.

```python
results = model(frame)
```

---

### Step 2: Tracking

ByteTrack generates temporary track IDs.

```python
tracker.update(detections)
```

---

### Step 3: Feature Extraction

Each crop is converted into a feature embedding.

```python
embedding = extract_embedding(person_crop)
```

---

### Step 4: Gallery Matching

Cosine similarity is computed against stored identities.

```python
similarity = cosine_similarity(
    query_embedding,
    gallery_embedding
)
```

---

### Step 5: Identity Assignment

```python
if similarity > 0.7:
    assign_existing_id()
else:
    create_new_identity()
```

---

# 📊 Performance Characteristics

| Metric               | Description             |
| -------------------- | ----------------------- |
| Detection FPS        | Depends on YOLO variant |
| Tracking Accuracy    | High with ByteTrack     |
| ReID Robustness      | Strong with OSNet       |
| Occlusion Recovery   | Excellent               |
| Identity Persistence | Long-term               |

---

# 🎯 Applications

### Smart Surveillance

Monitor individuals across large areas.

### Retail Analytics

Customer movement tracking.

### Airport Security

Identity persistence across cameras.

### Smart Cities

Pedestrian flow monitoring.

### Parking Management

Vehicle and person tracking.

### Crowd Analysis

Behavior and trajectory understanding.

---

# 🔮 Future Improvements

* Multi-camera tracking
* Cross-camera ReID
* DeepSORT integration
* StrongSORT support
* FastReID implementation
* TensorRT optimization
* Edge deployment
* Real-time dashboard

---

# 📈 Sample Output

```text
Frame 120:
Person A → ID 3

Occlusion:
Person disappears

Frame 180:
Person reappears

ReID Match:
Cosine Similarity = 0.87

Assigned:
ID 3 (Recovered)
```

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push branch
5. Open Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Aravind Pyli**

AI Researcher | Computer Vision Engineer | Machine Learning Enthusiast

### Research Interests

* Multi-Object Tracking
* Person Re-Identification
* Video Analytics
* Edge AI
* Computer Vision
* Deep Learning

---

⭐ If you found this project useful, consider giving the repository a star.
