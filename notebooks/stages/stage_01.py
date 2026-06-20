# Stage 1: chore: initialize Colab notebook with dependency installation
# ==================================================

#@title 1️⃣ Setup and Install Dependencies

!pip install ultralytics torchreid supervision==0.1.0

!git clone https://github.com/ifzhang/ByteTrack.git
%cd ByteTrack
!sed -i 's/onnx==1.8.1/onnx==1.9.0/g' requirements.txt
!pip install -r requirements.txt
!pip install cython_bbox onemetric loguru lap thop
!python3 setup.py develop
%cd ..

import sys
sys.path.append('./ByteTrack')

print("Setup done.")