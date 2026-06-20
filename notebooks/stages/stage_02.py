# Stage 2: chore: clone ByteTrack repo and patch onnx version requirement
# ==================================================

import inspect
from yolox.tracker.byte_tracker import BYTETracker
print(inspect.signature(BYTETracker.__init__))