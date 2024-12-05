import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('cifar10_model.h5')

# Class names for CIFAR-10 dataset
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Title for Streamlit app
st.title('CIFAR-10 Image Classifier')

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    img = image.load_img(uploaded_file, target_size=(32, 32))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for prediction
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # Display the prediction
    st.write(f"Prediction: {cifar10_classes[predicted_class]}")
    st.write(f"Confidence: {predictions[0][predicted_class]:.2f}")

# Footer with "Prepared By Prashant Kumar"
st.write("Prepared By Prashant Kumar")
