import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

pkl_path = 'intel_image.h5'

# loading the model
model = load_model(pkl_path)

# Name of labels
labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

with st.sidebar:
    st.subheader('Available Labels')
    st.markdown('- buildings')
    st.markdown('- forest')
    st.markdown('- glacier')
    st.markdown('- mountain')
    st.markdown('- sea')
    st.markdown('- street')

# Setting the title of app
st.header("Nature Vision")

st.image(
    'https://user-images.githubusercontent.com/96771321/214588217-b037c3e3-bbb3-4e52-9da7-3459cbdc27b4.jpg'
)

# Uploading the dog image
nature_image = st.file_uploader("Upload image", type="jpg")
submit = st.button("Predict")

# On Clicking Predict Button
if submit:
    if nature_image is not None:
        # Convert the file to an open cv image
        file_bytes = np.asarray(bytearray(nature_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Resizing the image
        opencv_image2 = cv2.resize(opencv_image, (150, 150))

        # Convert image to 4 Dimensions
        opencv_image2 = image.img_to_array(opencv_image2)
        opencv_image2 = np.expand_dims(opencv_image2.copy(), axis=0)
        opencv_image2 = opencv_image2 / 255.0

        # Make Prediction
        prediction = model.predict(opencv_image2)

        st.title(str("The is a " + labels[np.argmax(prediction)]))

        # Display the image
        st.image(opencv_image, channels="BGR")