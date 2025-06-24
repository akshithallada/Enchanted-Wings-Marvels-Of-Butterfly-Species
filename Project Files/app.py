# app.py (Flask backend)
import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
import shutil
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from bs4 import BeautifulSoup

app = Flask(__name__)

# Paths
MODEL_PATH = "best_model_mobilenet.keras"
ENCODER_PATH = "label_encoder.pkl"
DATASET_PATH = "dataset/train"
CSV_PATH = "dataset/Training_set.csv"

# Load model and encoder once
model = load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
df = pd.read_csv(CSV_PATH)
df['file_path'] = df['filename'].apply(lambda x: os.path.join(DATASET_PATH, x))

# Utility: Fetch info from Wikipedia

def fetch_species_info(species_name):
    try:
        search_term = species_name.replace(" ", "_") + "_(butterfly)"
        url = f"https://en.wikipedia.org/wiki/{search_term}"
        response = requests.get(url, timeout=5)
        if response.status_code == 404:
            # fallback
            search_term = species_name.replace(" ", "_")
            url = f"https://en.wikipedia.org/wiki/{search_term}"
            response = requests.get(url, timeout=5)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = soup.find_all("p")
            for p in paragraphs:
                text = p.get_text().strip()
                if len(text) > 100 and any(k in text.lower() for k in ["butterfly", "species", "family"]):
                    cleaned = ''.join(text.split("[")[0])  # Remove [1], [2], etc.
                    return cleaned
        return "No informative content found."
    except:
        return "No informative content found."

# Utility: Predict uploaded image

def predict_species(image_path):
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    species = label_encoder.inverse_transform([class_index])[0]
    return species

# Utility: Copy one image from dataset to static

def get_image_path(species_name):
    entry = df[df['label'].str.lower() == species_name.lower()]
    if not entry.empty:
        dataset_path = entry.iloc[0]['file_path']
        static_path = os.path.join("static", os.path.basename(dataset_path))

        if not os.path.exists(static_path):
            try:
                shutil.copy(dataset_path, static_path)
            except Exception as e:
                print(f"Failed to copy image: {e}")
                return None
        return os.path.basename(static_path)  # Return relative to /static
    return None

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/search', methods=['POST'])
def search():
    species = request.form['species_name']
    image_path = get_image_path(species)
    description = fetch_species_info(species)
    return render_template("result.html", label=species, image=image_path, description=description)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if file:
        filepath = os.path.join("static", file.filename)
        file.save(filepath)
        predicted_species = predict_species(filepath)
        image_path = get_image_path(predicted_species)
        description = fetch_species_info(predicted_species)
        return render_template("result.html", label=predicted_species, image=image_path, description=description)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)