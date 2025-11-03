import datasets
import numpy as np
from PIL import Image
from itertools import islice
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# >>> usa TF Keras, non "keras" puro
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def _pil_to_rgb_array(pil_img, img_size):
    if not isinstance(pil_img, Image.Image):
        pil_img = Image.fromarray(np.array(pil_img))
    pil_img = pil_img.convert("RGB").resize(img_size)
    return np.asarray(pil_img, dtype=np.float32) / 255.0  # (H,W,3)

def get_dataset(
    name="cats_vs_dogs",
    img_size=(224, 224),
    batch_size=32,
    val_split=0.2,
    debug_samples=None,    # se impostato, carica pochi esempi via streaming
    shuffle_buffer=None
):
    if debug_samples is not None:
        # ---- DEBUG: streaming, scarica solo ciò che consumi ----
        if shuffle_buffer is None:
            shuffle_buffer = max(50, min(5 * debug_samples, 500))  # es. 10 -> 50
        print(f"⚙️ DEBUG MODE: streaming, prendo {debug_samples} esempi (buffer={shuffle_buffer})")

        ds_stream = datasets.load_dataset(name, split="train", streaming=True)
        ds_stream = ds_stream.shuffle(seed=42, buffer_size=shuffle_buffer)

        images, labels = [], []
        for ex in islice(ds_stream, debug_samples):
            try:
                arr = _pil_to_rgb_array(ex["image"], img_size)
                lbl = int(ex["labels"])  # >>> chiave 'labels' (plurale)
            except Exception:
                continue
            if arr.ndim != 3 or arr.shape[2] != 3:
                continue
            images.append(arr); labels.append(lbl)

        if not images:
            raise RuntimeError("Nessun esempio valido ottenuto in streaming.")
        images = np.stack(images)
        labels = np.array(labels, dtype=np.int64)

    else:
        # ---- FULL: caricamento completo ----
        ds = datasets.load_dataset(name, split="train")
        def _pp(example):
            return {"image": _pil_to_rgb_array(example["image"], img_size),
                    "labels": int(example["labels"])}  # >>> mantieni 'labels'
        ds = ds.map(_pp, remove_columns=ds.column_names)
        images = np.stack(ds["image"])
        labels = np.array(ds["labels"], dtype=np.int64)

    # split train/val (tollerante ai subset piccoli/mono-classe)
    if len(np.unique(labels)) > 1 and len(labels) >= 5:
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=val_split, random_state=42, stratify=labels
        )
    else:
        X_train = X_val = images
        y_train = y_val = labels

    # generatori (TF Keras)
    train_datagen = ImageDataGenerator()
    val_datagen   = ImageDataGenerator()

    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_generator   = val_datagen.flow(X_val,   y_val,   batch_size=batch_size, shuffle=False)
    return train_generator, val_generator

def visualize_samples(generator, class_names=None, num_samples=5):
    images, labels = next(generator)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    for i in range(min(num_samples, len(images))):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i])
        idx = int(labels[i]) if np.ndim(labels[i]) == 0 else int(np.argmax(labels[i]))
        title = class_names[idx] if class_names else str(idx)
        plt.title(title); plt.axis("off")
    plt.show()