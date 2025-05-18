from flask import Flask, request, jsonify
import torch
from PIL import Image
import io

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
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

    results = model(img, size=640)
    results_json = results.pandas().xyxy[0].to_dict(orient='records')

    return jsonify({'predictions': results_json})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
