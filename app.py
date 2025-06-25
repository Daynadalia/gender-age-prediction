import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the trained Keras model
@st.cache_resource
def load_trained_model():
    model = load_model("model.h5")
    return model

model = load_trained_model()

def preprocess_image(image_file):
    image = Image.open(image_file).convert("L")  # Convert to grayscale
    image = image.resize((128, 128))             # Resize to match model input
    img_array = np.array(image) / 255.0          # Normalize
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dim (H, W, 1)
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dim (1, H, W, 1)
    return img_array

# Prediction function
def predict(image_array):
    gender_pred, age_pred = model.predict(image_array)
    
    gender = "Male" if np.argmax(gender_pred) == 1 else "Female"
    age = int(age_pred[0][0])
    
    return gender, age

# Streamlit UI
st.title("Gender & Age Prediction from Image (Keras Model)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Predicting..."):
        image_array = preprocess_image(uploaded_file)
        gender, age = predict(image_array)
    
    st.success(f"**Predicted Gender:** {gender}")
    st.success(f"**Predicted Age:** {age} years")
