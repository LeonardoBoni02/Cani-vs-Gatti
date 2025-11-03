import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, Response

# Config base
CLASS_NAMES = ["Cat", "Dog"]
IMG_SIZE = (224, 224)
MODEL_PATH = os.environ.get("MODEL_PATH", "model/cnn_cats_vs_dogs.keras")

# Carica il modello una volta (all'avvio)
import keras
MODEL = keras.models.load_model(MODEL_PATH)

app = Flask(__name__)

HTML_FORM = """
<!doctype html>
<title>Cats vs Dogs - Demo</title>
<h1>Cats vs Dogs - Upload an image</h1>
<form method="POST" action="/predict" enctype="multipart/form-data">
  <input type="file" name="image" accept="image/*" required>
  <button type="submit">Predict</button>
</form>
<p>Oppure usa: <code>curl -F "image=@path/to/img.jpg" http://localhost:8080/predict</code></p>
"""

@app.route("/", methods=["GET"])
def index():
    return Response(HTML_FORM, mimetype="text/html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

def preprocess(pil_image, img_size=IMG_SIZE):
    # converte a RGB, ridimensiona, normalizza [0,1], aggiunge batch
    img = pil_image.convert("RGB").resize(img_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file field"}), 400
    file = request.files["image"]
    try:
        pil_image = Image.open(io.BytesIO(file.read()))
    except Exception:
        return jsonify({"error": "invalid image file"}), 400

    x = preprocess(pil_image)
    preds = MODEL.predict(x)
    # Se il tuo modello usa softmax a 2 neuroni:
    if preds.shape[-1] == 2:
        probs = preds[0].tolist()
        idx = int(np.argmax(preds[0]))
    else:
        # Caso binario con sigmoid (1 neurone): p(dog) = preds[0][0], p(cat)=1-p
        p_dog = float(preds[0][0])
        probs = [1.0 - p_dog, p_dog]
        idx = int(p_dog >= 0.5)

    result = {
        "predicted_class": CLASS_NAMES[idx],
        "probabilities": {CLASS_NAMES[0]: float(probs[0]), CLASS_NAMES[1]: float(probs[1])}
    }
    return jsonify(result)

if __name__ == "__main__":
    # Per debug locale (in Docker usiamo gunicorn)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
