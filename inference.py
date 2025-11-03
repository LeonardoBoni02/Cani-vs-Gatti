CLASS_NAMES = ["Cat", "Dog"]

def infer_image(image_path, model_path='cnn_cats_vs_dogs.keras', img_size=(224, 224)):
    import keras, numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    # Carica il modello
    model = keras.models.load_model(model_path)

    # Carica e prepara l'immagine
    img = Image.open(image_path).convert("RGB").resize(img_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape (1, 224, 224, 3)

    # Inferenza
    preds = model.predict(arr)
    probs = preds[0]  # vettore di probabilità, es. [0.1, 0.9]
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]

    # Stampa a video risultati
    print(f"Predizione: {label}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {probs[i]*100:.2f}%")

    # Mostra immagine con label predetta
    plt.imshow(img)
    plt.title(f"{label} ({probs[idx]*100:.1f}%)")
    plt.axis("off")
    plt.show()

    return label
