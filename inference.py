CLASS_NAMES = ["Cat", "Dog"]

def infer_image(image_path, model_path='cnn_cats_vs_dogs.keras', img_size=(224, 224)):
    import keras, numpy as np
    from PIL import Image

    model = keras.models.load_model(model_path)
    img = Image.open(image_path).convert("RGB").resize(img_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)
    idx = int(np.argmax(preds, axis=1)[0])
    return CLASS_NAMES[idx]  # <-- restituisce "Cat" o "Dog"