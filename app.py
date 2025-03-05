from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
MODEL_PATH = "model.h5"
model = load_model(MODEL_PATH)
labels = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Preprocess the image
    img = cv2.imread(filepath)
    if img is None:
        return jsonify({"error": "Invalid image format"}), 400
    
    img = cv2.resize(img, (150, 150))
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions) * 100)
    
    return jsonify({"prediction": labels[predicted_class], "confidence": confidence})

if __name__ == '__main__':
    app.run(debug=True)
