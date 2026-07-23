from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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


def load_data():
    """Load and normalize the Fashion-MNIST dataset."""

    fashion_mnist = keras.datasets.fashion_mnist

    (x_train, y_train), (x_test, y_test) = (
        fashion_mnist.load_data()
    )

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension for CNN input.
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, y_train, x_test, y_test


def build_model():
    """Build a simple convolutional neural network."""

    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),

            layers.Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu",
            ),

            layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            layers.Conv2D(
                64,
                kernel_size=(3, 3),
                activation="relu",
            ),

            layers.MaxPooling2D(
                pool_size=(2, 2)
            ),

            layers.Flatten(),

            layers.Dropout(0.30),

            layers.Dense(
                128,
                activation="relu",
            ),

            layers.Dense(
                10,
                activation="softmax",
            ),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_training_chart(history):
    """Save the training and validation accuracy chart."""

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy",
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Training Performance")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        CHART_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def train_model():
    """Train, evaluate, and save the CNN model."""

    x_train, y_train, x_test, y_test = load_data()

    model = build_model()

    model.summary()

    history = model.fit(
        x_train,
        y_train,
        epochs=8,
        batch_size=64,
        validation_split=0.20,
    )

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=0,
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(MODEL_PATH)

    save_training_chart(history)

    predictions = model.predict(
        x_test[:10],
        verbose=0,
    )

    print("\nSample Predictions:")

    for index, prediction in enumerate(predictions):
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        actual_class = int(y_test[index])

        print(
            f"Image {index + 1}: "
            f"Predicted={CLASS_NAMES[predicted_class]}, "
            f"Actual={CLASS_NAMES[actual_class]}, "
            f"Confidence={confidence * 100:.2f}%"
        )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Training chart saved to: {CHART_PATH}")


if __name__ == "__main__":
    train_model()