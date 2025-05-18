from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import os
import sys

# Add yolov5 path
YOLOV5_PATH = os.path.join(os.getcwd(), 'yolov5')
sys.path.append(YOLOV5_PATH)

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox
import numpy as np

app = Flask(__name__)
device = select_device('cpu')  # Safe for Render (no GPU)
model = DetectMultiBackend('best.pt', device=device)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = np.array(img)
    img = letterbox(img, 640, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0).to(device)

    pred = model(img)
    pred = non_max_suppression(pred, 0.25, 0.45)[0]

    results = []
    for *xyxy, conf, cls in pred:
        results.append({
            'bbox': [round(x.item(), 2) for x in xyxy],
            'confidence': round(conf.item(), 2),
            'class': int(cls.item())
        })

    return jsonify({'predictions': results})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
