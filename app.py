from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import os
import sys

app = Flask(__name__)

# Set path to YOLOv5 repo
YOLOV5_PATH = os.path.join(os.getcwd(), 'yolov5')
sys.path.append(YOLOV5_PATH)

# Import detection logic
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox

device = select_device('')
model = DetectMultiBackend('best.pt', device=device)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception:
        return jsonify({'error': 'Invalid image format'}), 400

    # Preprocess image
    img = letterbox(img, 640, stride=32, auto=True)[0]
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0).to(device)

    # Inference
    pred = model(img)
    pred = non_max_suppression(pred, 0.25, 0.45)[0]

    # Post-process
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
