import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load model
model = load_model("cnn_cifar10_v2.keras", compile=False, custom_objects={})
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

st.title("Image Classification App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).resize((32,32))
    img_array = img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.success(f"Predicted Class: {predicted_class}")


st.bar_chart(prediction[0])