import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import json
import os
import time
import pickle   
# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Flower Identifier AI",
    page_icon="🌸",
    layout="centered"
)

# ---------------- CONFIG ----------------
MODEL_PATH = "flower_species_model (1).keras"
DICT_PATH = "flower_info.json"
CLASS_NAMES_PATH = "class_names.pkl"
IMG_SIZE_PATH = "img_size.pkl"
# Load class names from pickle file
with open(CLASS_NAMES_PATH, "rb") as f:
    CLASS_NAMES = pickle.load(f)

# Load image size from pickle file
with open(IMG_SIZE_PATH, "rb") as f:
    IMG_SIZE = pickle.load(f)

# ---------------- LOAD MODEL (CACHED + WARMUP) ----------------
@st.cache_resource(show_spinner="🔄 Loading AI model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found")
        return None

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # 🔥 Warm-up inference (VERY IMPORTANT)
    dummy_input = np.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
    model.predict(dummy_input)

    return model


# ---------------- LOAD FLOWER INFO ----------------
@st.cache_data
def load_flower_info():
    if not os.path.exists(DICT_PATH):
        st.error("❌ flower_info.json not found")
        return {}

    with open(DICT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------- LOAD RESOURCES ----------------
model = load_model()
flower_info = load_flower_info()

# ---------------- UI ----------------
st.title("🌸 Flower Identifier AI")
st.write("Upload a flower image to identify its species and care information.")

uploaded_file = st.file_uploader(
    "📷 Upload flower image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------
if uploaded_file and model:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        # -------- Preprocess --------
        start = time.time()

        img = ImageOps.fit(
            image,
            IMG_SIZE,
            Image.Resampling.LANCZOS
        )

        img_array = np.asarray(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # -------- Predict --------
        with st.spinner("🔍 Analyzing flower..."):
            preds = model.predict(img_array)

        prediction_time = time.time() - start

        idx = int(np.argmax(preds))
        confidence = float(preds[0][idx])
        label = CLASS_NAMES[idx]

        # -------- Results --------
        st.subheader("🌼 Prediction Result")
        st.write(f"**Species:** {label.title()}")
        st.write(f"**Confidence:** {confidence:.2%}")
        st.caption(f"⏱ Prediction time: {prediction_time:.2f} seconds")

        if confidence < 0.60:
            st.warning("⚠️ Low confidence prediction")

# ---------------- INFO SECTION ----------------
if uploaded_file and model:
    st.divider()

    info = flower_info.get(label, {})

    if info:
        st.header(info.get("name", label.title()))
        st.write(info.get("description", "No description available."))

        c1, c2 = st.columns(2)

        with c1:
            st.info(
                f"🌱 **Care Instructions**\n\n"
                f"{info.get('care_instructions', 'Not available')}"
            )

        with c2:
            tox = info.get("toxicity", "Unknown")
            if "non" in tox.lower():
                st.success(f"🐾 **Toxicity**\n\n{tox}")
            else:
                st.error(f"☠️ **Toxicity**\n\n{tox}")

    else:
        st.info("No additional information found.")

# ---------------- FOOTER ----------------
st.markdown("---")
