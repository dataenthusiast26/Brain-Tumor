# 🧠 Brain Tumor Detection API with Grad-CAM & Gemini AI

This project is a powerful deep learning-based web API for **brain tumor detection from MRI images**, integrated with **Grad-CAM visualization** and **AI-generated medical reports** using Google's **Gemini API**.

---

## 🚀 Features

- ✅ **Brain Tumor Classification** (Glioma, Meningioma, Pituitary, No Tumor)
- 🔍 **Grad-CAM Visualization** for model explainability
- 🤖 **Gemini API** for AI-generated medical summaries
- 🧠 **Custom CNN model** trained on brain MRI dataset
- 🧾 **Prediction History** endpoint
- 💬 **AI Chatbot** endpoint (for general medical Q&A)

---

## 🧩 Tech Stack

- **FastAPI** – Backend API framework
- **TensorFlow / Keras** – Deep learning model
- **Pillow & NumPy** – Image processing
- **Matplotlib & OpenCV** – Grad-CAM heatmaps
- **Gemini API** – Google’s GenAI for reports
- **Python 3.10+**

---

## 📁 Dataset Used

- **Brain MRI Dataset**  
  Format: 128x128 RGB MRI scans  
  Classes:  
  - Glioma Tumor  
  - Meningioma Tumor  
  - Pituitary Tumor  
  - No Tumor  

> *(Use your own trained `mri_model.h5` with CNN architecture matching input shape `(128, 128, 3)`)*

---

## 🔧 API Endpoints

### 🔹 `/predict` [POST]

Upload a brain MRI image and receive:

- Predicted Tumor Class
- Confidence Score
- Class Probabilities
- Gemini AI Medical Report

**Request:**
```bash
curl -X POST -F "file=@sample_mri.jpg" http://localhost:8000/predict
