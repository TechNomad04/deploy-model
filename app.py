import torch
from PIL import Image
import numpy as np
import gradio as gr
import os
import sys

# Path to YOLOv5 repo (ensure yolov5/ is in the same directory)
YOLOV5_PATH = os.path.join(os.getcwd(), 'yolov5')
sys.path.append(YOLOV5_PATH)

# Import YOLOv5 internals
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox
import cv2

# Load model
device = select_device('')
model = DetectMultiBackend('best.pt', device=device)
model.eval()

# Class labels (optional: load from model.names if needed)
CLASS_NAMES = model.names if hasattr(model, 'names') else {}

def detect_objects(image: Image.Image):
    orig_img = np.array(image)

    # Preprocess
    img = letterbox(orig_img, 640, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img)
        pred = non_max_suppression(pred, 0.25, 0.45)[0]

    # Draw and extract results
    results = []
    if pred is not None and len(pred):
        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = map(int, xyxy)
            label = CLASS_NAMES[int(cls)] if CLASS_NAMES else str(int(cls))
            confidence = round(conf.item(), 2)
            results.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "class": label
            })
            # Draw on image
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(orig_img, f'{label} {confidence}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return Image.fromarray(orig_img), results

# Gradio interface
gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(label="Detected Image"), gr.JSON(label="Predictions")],
    title="YOLOv5 Object Detection",
    description="Upload an image to detect objects using a custom YOLOv5 model."
).launch()
