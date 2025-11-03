# Immagine base Python slim
FROM python:3.11-slim

# Evita bytecode e forza stdout/stderr “unbuffered”
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Installa dipendenze di sistema minime per Pillow e TF
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg62-turbo-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# Copia requirements e installa
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia codice e modello
COPY app.py ./
COPY cnn_cats_vs_dogs.keras ./

# Espone porta
EXPOSE 8080

# Avvio con gunicorn (1 worker sync per MVP; puoi aumentare)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
