import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_detector_model():
    return load_model("fake_image_detector.h5")

model = load_detector_model()

# App UI
st.set_page_config(page_title="Fake Face Detector", layout="centered")
st.title("🧠 Fake Faces Detector")
st.write("Upload a face image and we'll tell you if it's real or fake.")

uploaded_file = st.file_uploader("📷 Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open and display image
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Preprocess image
        img = img.resize((128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)[0][0]

        # Output result
        if prediction > 0.5:
            st.success(f"✅ Predicted: Real ({round(prediction * 100, 2)}% confidence)")
        else:
            st.error(f"❌ Predicted: Fake ({round((1 - prediction) * 100, 2)}% confidence)")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
