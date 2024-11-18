import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2

# Load the ONNX model
@st.cache_resource
def load_model(model_path):
    session = ort.InferenceSession(model_path)
    return session

# Preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize the image to match model input
    image = np.array(image).astype(np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.transpose(image, (0, 3, 1, 2))  # Rearrange dimensions for model
    return image

# Make a prediction
def predict(image, model_session):
    input_name = model_session.get_inputs()[0].name
    output = model_session.run(None, {input_name: image})
    return output

# Streamlit app interface
st.title("Brain Tumor Classification")

# Upload image
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess and predict
    model_path = "your_model.onnx"  #provide path to the model that gets saved after running the ipynb file.
    session = load_model(model_path)
    processed_image = preprocess_image(image)
    prediction = predict(processed_image, session)

    # Display result
    predicted_class = np.argmax(prediction)  # Assumes model outputs probabilities
    st.write(f"Predicted Class: {'Brain Tumor' if predicted_class == 1 else 'No Tumor'}")
