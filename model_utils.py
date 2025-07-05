import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained Keras model
def load_keras_model(model_path="model.h5"):
    model = load_model(model_path)
    return model

# Preprocess the uploaded image
def preprocess_image(image_file, target_size=(224, 224)):
    image = Image.open(image_file).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Predict gender and age
def predict(model, image_array):
    gender_output, age_output = model.predict(image_array)
    gender = "Male" if gender_output[0][0] >= 0.5 else "Female"
    age = int(age_output[0][0])
    return gender, age
