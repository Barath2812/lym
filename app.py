# =========================================
# LYMPHOMA CANCER DETECTION — TFLITE
# =========================================

import streamlit as st
import numpy as np
import cv2
import json
from PIL import Image
import tensorflow as tf

# ---------------- CONFIG ----------------
IMG_SIZE = 256
MODEL_PATH = "lymphoma_final_model.tflite"
CLASS_PATH = "class_names.json"

st.set_page_config(
    page_title="Lymphoma Cancer Detection",
    page_icon="🧪",
    layout="centered"
)

# ---------------- LOAD TFLITE MODEL ----------------
@st.cache_resource
def load_tflite_and_classes():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    with open(CLASS_PATH, "r") as f:
        class_names = json.load(f)

    return interpreter, class_names

interpreter, class_names = load_tflite_and_classes()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# EfficientNet preprocessing
preprocess = tf.keras.applications.efficientnet.preprocess_input

# ---------------- UI ----------------
st.title("🧪 Lymphoma Cancer Detection System")
st.caption("EfficientNet-based histopathology image classification")

st.divider()

uploaded_file = st.file_uploader(
    "📤 Upload Histopathology Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    if st.button("🔬 Predict Lymphoma Type"):
        with st.spinner("Analyzing image…"):
            img = np.array(image)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = np.expand_dims(img, axis=0)
            img = preprocess(img).astype(np.float32)

            interpreter.set_tensor(input_details[0]["index"], img)
            interpreter.invoke()

            preds = interpreter.get_tensor(output_details[0]["index"])[0]
            idx = int(np.argmax(preds))
            confidence = float(preds[idx]) * 100

            st.success(f"🧪 Prediction: **{class_names[idx]}**")
            st.progress(int(confidence))
            st.info(f"Confidence: **{confidence:.2f}%**")

st.caption("⚠️ For research and educational purposes only")
