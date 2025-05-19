from ultralytics import YOLO
from flask import Flask, request, jsonify
import numpy as np
import cv2

model = YOLO('best.pt')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify ({'no image part in the request'}),400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'no file selected'}),400
    
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = model(img)

    detections = []
    for r in results :
        b = r.boxes[0]
        cls_id = int(b.cls)
        conf = float(b.conf)
        coords = b.xyxy[0].tolist()
        label = r.names[cls_id]
        detections.append({
            'class': label,
            'confidence': round(conf, 3),
            'box': [round(c, 2) for c in coords]
        })

    return jsonify({'detections' : detections})

if __name__ == '__main__':
    app.run(debug=True)


