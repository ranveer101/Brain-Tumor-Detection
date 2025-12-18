import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
IMAGE_SIZE = 128

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model("brain_tumor_model.keras")

# 🔥 AUTO-DETECT CLASS ORDER (NO JSON FILE)
TRAIN_DIR = "MRI_images/Training"
CLASS_NAMES = sorted([
    folder for folder in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, folder))
])

print("Detected class order:", CLASS_NAMES)

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)

    img = image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    class_index = int(np.argmax(preds))

    result = CLASS_NAMES[class_index]
    confidence = float(preds[class_index] * 100)

    return jsonify({
        "result": result.replace("_", " ").title(),
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
IMAGE_SIZE = 128

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model("brain_tumor_model.keras")

# 🔥 AUTO-DETECT CLASS ORDER (NO JSON FILE)
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]


print("Detected class order:", CLASS_NAMES)

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)

    img = image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    class_index = int(np.argmax(preds))

    result = CLASS_NAMES[class_index]
    confidence = float(preds[class_index] * 100)

    return jsonify({
        "result": result.replace("_", " ").title(),
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

