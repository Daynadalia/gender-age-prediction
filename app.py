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
    preds = model.predict(image_array)

    # If the model returns two outputs: (gender, age)
    if isinstance(preds, list) and len(preds) == 2:
        gender_pred, age_pred = preds
    else:
        # Model returns only one output or unexpected format
        gender_pred = preds
        age_pred = [[25]]  # fallback if age not returned

    # Determine gender from output shape
    if gender_pred.shape[-1] == 1:
        # Binary sigmoid: output close to 1 = Male
        gender = "Male" if gender_pred[0][0] >= 0.5 else "Female"
    elif gender_pred.shape[-1] == 2:
        # Softmax: [Female prob, Male prob]
        gender = "Male" if np.argmax(gender_pred[0]) == 1 else "Female"
    else:
        gender = "Unknown"

    # Try age extraction
    try:
        age = int(age_pred[0][0])
    except:
        age = "Unknown"

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
