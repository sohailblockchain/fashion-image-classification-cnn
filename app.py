from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


MODEL_PATH = Path("models/fashion_cnn.keras")
CHART_PATH = Path("training_history.png")

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


st.set_page_config(
    page_title="Fashion Image Classifier",
    page_icon="👕",
    layout="centered",
)

st.title("Fashion Image Classification using CNN")

st.write(
    "Upload a clothing image and the deep learning model "
    "will predict its category."
)

if not MODEL_PATH.exists():
    st.error(
        "Model not found. Run `python train_model.py` first."
    )
    st.stop()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

uploaded_file = st.file_uploader(
    "Upload a clothing image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")

    st.image(
        image,
        caption="Uploaded Image",
        width=250,
    )

    # Resize image to Fashion-MNIST dimensions.
    resized_image = ImageOps.fit(
        image,
        (28, 28),
    )

    image_array = np.array(
        resized_image
    ).astype("float32")

    # Fashion-MNIST images generally use a dark background.
    if image_array.mean() > 127:
        image_array = 255 - image_array

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=(0, -1),
    )

    prediction = model.predict(
        image_array,
        verbose=0,
    )[0]

    predicted_index = int(
        np.argmax(prediction)
    )

    confidence = float(
        np.max(prediction)
    )

    st.subheader("Prediction")

    st.success(
        f"Predicted Category: "
        f"{CLASS_NAMES[predicted_index]}"
    )

    st.metric(
        "Confidence",
        f"{confidence * 100:.2f}%",
    )

    st.write("Class Probabilities")

    probability_data = {
        CLASS_NAMES[index]: float(
            prediction[index]
        )
        for index in range(
            len(CLASS_NAMES)
        )
    }

    st.bar_chart(probability_data)

st.divider()

st.subheader("Training Performance")

if CHART_PATH.exists():
    st.image(
        str(CHART_PATH),
        caption=(
            "Training and validation accuracy "
            "across epochs."
        ),
        use_container_width=True,
    )