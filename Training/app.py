from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
import os

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Load the trained model
model = load_model("model.h5")

# Class labels
labels = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# Ensure the uploads folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(img, (150, 150))
    img = img.astype("float32") / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Route for handling the web UI and prediction
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)  # Save uploaded file

        # Preprocess and make a prediction
        img = preprocess_image(file_path)
        if img is None:
            return render_template("index.html", error="Invalid image format")

        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions)) * 100  # Convert to percentage

        return render_template(
            "index.html",
            result=labels[predicted_class],
            confidence=f"{confidence:.2f}%",
            file_path=file.filename,  # Only pass the filename for display
        )

    return render_template("index.html", result=None)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
