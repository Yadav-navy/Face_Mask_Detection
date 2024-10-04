import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('my_model.h5')

# Define a function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to the input size expected by the model
    image = image.convert('RGB')  # Convert to RGB
    image = np.array(image)  # Convert to numpy array
    image_scaled = image / 255.0  # Normalize
    image_reshaped = np.reshape(image_scaled, [1, 128, 128, 3])  # Add batch dimension
    return image_reshaped

# Define a function to get prediction from the model
def predict_mask(image_array):
    predictions = model.predict(image_array)
    pred_label = np.argmax(predictions)
    return pred_label, predictions[0]

# Streamlit app
st.markdown(
    """
    <style>
    .main {
        background-color: #82a8cd ;
    }
    .title {
        
        color: black;
        text-align: center;
        font-size: 3em; 
    }
    </style>
    """,

    unsafe_allow_html=True
)

st.markdown(
    "<h1 class='title'>Mask Vision</h1>",
    unsafe_allow_html=True
)
# Centered and bold text
st.markdown(
    "<h5 style='color: #000080; text-align: center; font-weight: bold;'>Upload an image to check if the person is wearing a face mask or not</h5>",
    unsafe_allow_html=True
)

uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess and predict
    image_array = preprocess_image(image)
    pred_label, probabilities = predict_mask(image_array)

    # Display prediction result
    if pred_label == 1:
        st.markdown(
            "<h2 style='color: green; text-align: center;'>The person in the image is wearing a mask.</h2>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<h2 style='color: red; text-align: center;'>The person in the image is not wearing a mask.</h2>",
            unsafe_allow_html=True
        )

    # Display probabilities with h5 heading and bold
    st.markdown(
        f"<h5 style='color: black; font-weight: bold;'>Chance of wearing a mask: {probabilities[1]:.2f}</h5>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<h5 style='color: black; font-weight: bold;'>Chance of not wearing a mask: {probabilities[0]:.2f}</h5>",
        unsafe_allow_html=True
    )
