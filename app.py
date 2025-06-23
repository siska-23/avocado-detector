from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from PIL import Image
import io
import os

app = Flask(__name__, static_folder='static')

# Load YOLOv5 model
model = YOLO("best.pt")

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')

    results = model(image)

    response = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            response.append({
                "kelas": model.names[cls_id],
                "confidence": round(conf, 2)
            })

    return jsonify({"prediksi": response})

if __name__ == '__main__':
    app.run(debug=True)