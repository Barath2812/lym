# =========================================
# LYMPHOMA CANCER DETECTION — STREAMLIT
# =========================================

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
from PIL import Image

# ---------------- CONFIG ----------------
IMG_SIZE = 256
MODEL_PATH = "lymphoma_final_model.keras"
CLASS_PATH = "class_names.json"

st.set_page_config(
    page_title="Lymphoma Cancer Detection",
    page_icon="🧪",
    layout="centered"
)

# ---------------- LOAD MODEL & CLASSES ----------------
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False
    )
    with open(CLASS_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names


model, class_names = load_model_and_classes()

# EfficientNet preprocessing (MUST match training)
preprocess = tf.keras.applications.efficientnet.preprocess_input

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f9fafc;
}
h1 {
    color: #2c2c2c;
}
.stButton > button {
    background-color: #6a1b9a;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-size: 16px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🧪 Lymphoma Cancer Detection System")
st.caption("AI-powered histopathology image classification using EfficientNet")

st.divider()

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload Histopathology Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    st.markdown("### 🔍 Analysis")

    if st.button("🔬 Predict Lymphoma Type"):
        with st.spinner("Analyzing image… Please wait"):
            try:
                # -------- PREPROCESS IMAGE --------
                img = np.array(image)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = np.expand_dims(img, axis=0)
                img = preprocess(img)   # 🔥 CRITICAL FIX

                # -------- PREDICTION --------
                preds = model.predict(img, verbose=0)[0]
                class_index = int(np.argmax(preds))
                confidence = float(preds[class_index]) * 100

                # -------- OUTPUT --------
                st.success(f"🧪 Prediction: **{class_names[class_index]}**")
                st.progress(int(confidence))
                st.info(f"Confidence: **{confidence:.2f}%**")

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

st.divider()
st.caption("⚠️ For research and educational purposes only")
