# 👕 Fashion Image Classification using CNN

An end-to-end Deep Learning project that classifies clothing images into one of 10 Fashion-MNIST categories using a Convolutional Neural Network (CNN). The project includes model training, evaluation, visualization, and an interactive Streamlit web application for real-time image classification.

---

## 🚀 Project Overview

This project demonstrates how Convolutional Neural Networks (CNNs) can classify grayscale fashion images using TensorFlow and Keras.

The model is trained on the Fashion-MNIST dataset and predicts one of the following clothing categories:

- T-shirt / Top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle Boot

---

## 📊 Model Performance

| Metric | Result |
|--------|---------|
| Test Accuracy | **90.85%** |
| Test Loss | **0.2492** |
| Final Validation Accuracy | **91.36%** |
| Final Training Accuracy | **91.99%** |

---

## 🧠 CNN Architecture

The model consists of:

- Input Layer (28 × 28 × 1)
- Conv2D (32 Filters)
- MaxPooling2D
- Conv2D (64 Filters)
- MaxPooling2D
- Flatten Layer
- Dropout (30%)
- Dense Layer (128 Neurons)
- Softmax Output Layer (10 Classes)

Total Parameters:

**225,034 Trainable Parameters**

---

## 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pillow
- Streamlit

---

## 📂 Project Structure

```text
fashion-image-classification-cnn/
│
├── models/
│   └── fashion_cnn.keras
│
├── app.py
├── train_model.py
├── training_history.png
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/sohailblockchain/fashion-image-classification-cnn.git

cd fashion-image-classification-cnn
```

Create Virtual Environment

```bash
python3 -m venv venv

source venv/bin/activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train_model.py
```

The trained model will be saved as:

```text
models/fashion_cnn.keras
```

---

## 🚀 Run Streamlit

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

## 📈 Features

- Image Upload
- CNN Prediction
- Confidence Score
- Probability Distribution
- Training Accuracy Graph
- Real-time Classification
- Interactive Streamlit UI

---

## 📚 Deep Learning Workflow

1. Load Fashion-MNIST Dataset
2. Normalize Images
3. Build CNN Architecture
4. Train the Model
5. Evaluate Model
6. Save Trained Model
7. Predict New Images
8. Deploy using Streamlit

---

## 📌 Future Improvements

- Transfer Learning (MobileNet / ResNet)
- Data Augmentation
- Confusion Matrix
- Precision & Recall per Class
- Model Explainability (Grad-CAM)
- Deploy on Streamlit Community Cloud

---

## 👨‍💻 Author

**Sohail Ahmed**

Senior Software Engineer | Blockchain Engineer | Data Science & AI Engineer

GitHub:
https://github.com/sohailblockchain

LinkedIn:
https://www.linkedin.com/in/sohail-ahmed-b40b66215/
