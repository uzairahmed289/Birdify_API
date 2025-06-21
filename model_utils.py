import tensorflow as tf
import numpy as np
from PIL import Image as PILImage
from rembg import remove
from PIL import Image as PILImage
import io
import os
from PIL import Image
from io import BytesIO
from rembg import remove

# Load your models
species_model = tf.keras.models.load_model('species_recognition_model.h5')
gender_model = tf.keras.models.load_model('gender_recognition_model.h5')

# Define your class labels
species_classes = {
    0: 'Cardinal',
    1: 'House Sparrow',
    # Add more class indices as needed
}

def preprocess_image(img_path, target_size=(224, 224)):
    img = PILImage.open(img_path).resize(target_size).convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_species(img_path):
    img_array = preprocess_image(img_path)
    predictions = species_model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return species_classes.get(predicted_class, f"Unknown ({predicted_class})")

def predict_gender(img_path):
    img_array = preprocess_image(img_path)
    prediction = gender_model.predict(img_array)
    return 'Male' if prediction > 0.5 else 'Female'

def remove_background_and_save(input_path, output_path):
    with open(input_path, "rb") as f:
        input_data = f.read()
    result = remove(input_data)

    # Convert result (with transparency) to white background
    with Image.open(BytesIO(result)).convert("RGBA") as im:
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])  # 3 is alpha channel
        bg.save(output_path, "JPEG")