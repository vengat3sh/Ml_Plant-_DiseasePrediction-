
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==================================================
# PROJECT PATHS
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "train")
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.keras")
HOME_IMAGE_PATH = os.path.join(BASE_DIR, "home_page.jpg")


# ==================================================
# LOAD CLASS NAMES
# ==================================================
@st.cache_data
def load_class_names():
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(
            f"Training folder not found: {TRAIN_DIR}"
        )

    # Folder names sorted alphabetically.
    # Must match the original training class order.
    class_names = sorted(
        folder
        for folder in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, folder))
    )

    if len(class_names) != 38:
        raise ValueError(
            f"Expected 38 class folders, found {len(class_names)}. "
            "Check your train directory."
        )

    return class_names


# ==================================================
# LOAD MODEL
# ==================================================
@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


# ==================================================
# IMAGE PREDICTION
# ==================================================
def model_prediction(uploaded_image):
    model = load_model()
    class_names = load_class_names()

    image = Image.open(uploaded_image).convert("RGB")
    image = image.resize((128, 128))

    # Keep preprocessing consistent with model training.
    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)
    predictions = np.asarray(predictions)

    if predictions.ndim != 2 or predictions.shape[0] != 1:
        raise ValueError(
            f"Unexpected model output shape: {predictions.shape}"
        )

    predictions = predictions[0]

    if len(predictions) != len(class_names):
        raise ValueError(
            f"Model outputs {len(predictions)} classes, but "
            f"{len(class_names)} class folders were found. "
            "Check the original training class mapping."
        )

    result_index = int(np.argmax(predictions))
    predicted_class = class_names[result_index]

    # Valid for a softmax probability output.
    confidence = float(predictions[result_index]) * 100

    return predicted_class, confidence


# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("Dashboard")

app_mode = st.sidebar.selectbox(
    "Select Page",
    ["Home", "About", "Disease Recognition"]
)


# ==================================================
# HOME PAGE
# ==================================================
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")

    if os.path.isfile(HOME_IMAGE_PATH):
        st.image(
            Image.open(HOME_IMAGE_PATH),
            caption="Plant Disease Recognition System",
            use_column_width=True
        )
    else:
        st.warning(
            "home_page.jpg not found. "
            "Place it inside the project folder if required."
        )

    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿

    Upload a plant leaf image to predict its possible disease.

    ### How It Works
    1. Open the Disease Recognition page.
    2. Upload a JPG, JPEG, or PNG leaf image.
    3. Click Predict.
    4. View the predicted class and model confidence.

    ### Important
    Predictions are model estimates and should be confirmed
    before making crop-treatment decisions.
    """)


# ==================================================
# ABOUT PAGE
# ==================================================
elif app_mode == "About":
    st.header("About the Dataset")

    st.markdown("""
    This application uses a trained deep-learning model
    to classify plant leaf images.
    """)

    try:
        class_names = load_class_names()

        st.write(f"**Number of classes:** {len(class_names)}")

        with st.expander("View training classes"):
            for index, class_name in enumerate(class_names):
                st.write(f"{index}: {class_name}")

    except Exception as error:
        st.error(f"Could not load class names: {error}")


# ==================================================
# DISEASE RECOGNITION PAGE
# ==================================================
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    uploaded_image = st.file_uploader(
        "Choose a plant leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is None:
        st.info("Please upload a plant leaf image.")

    else:
        try:
            image = Image.open(uploaded_image).convert("RGB")

            st.image(
                image,
                caption="Uploaded Plant Image",
                use_column_width=True
            )

            if st.button("Predict", type="primary"):
                with st.spinner("Analyzing image..."):
                    uploaded_image.seek(0)

                    predicted_class, confidence = model_prediction(
                        uploaded_image
                    )

                st.success(f"Prediction: {predicted_class}")
                st.metric(
                    "Model confidence",
                    f"{confidence:.2f}%"
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
