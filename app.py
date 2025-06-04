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

# Preprocess the uploaded image
def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image = image.resize((224, 224))  # Change if your model uses a different input size
    img_array = np.array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
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
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Predicting..."):
        image_array = preprocess_image(uploaded_file)
        gender, age = predict(image_array)
    
    st.success(f"**Predicted Gender:** {gender}")
    st.success(f"**Predicted Age:** {age} years")
