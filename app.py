import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "uploads"
IMAGE_SIZE = 128

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = load_model("brain_tumor_model.keras")

# ✅ Class names (must match training order)
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        secure_filename(file.filename)
    )
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)[0]
    class_index = int(np.argmax(preds))

    return jsonify({
        "result": CLASS_NAMES[class_index].replace("_", " ").title(),
        "confidence": round(float(preds[class_index]) * 100, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
