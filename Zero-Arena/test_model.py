

import torch

# Patch torch.load to always allow full objects
_orig_load = torch.load
torch.load = lambda *args, **kwargs: _orig_load(*args, weights_only=False, **kwargs)

from ultralyticsplus import YOLO, render_result
import cv2

# Load model
model = YOLO('home/dijo404/git/Zero-Arena/yolov8s.pt')

# Test image
image_path = "/home/dijo404/Downloads/test_1.jpg"
image = cv2.imread(image_path)

results = model(image)
annotated = results[0].plot()

cv2.imshow("Retail Wizardry", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
