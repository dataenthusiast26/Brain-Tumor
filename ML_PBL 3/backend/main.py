from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import requests

app = FastAPI()

# ✅ Load trained model
MODEL_PATH = "mri_model.h5"
try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    raise HTTPException(status_code=500, detail="Model loading failed")

# ✅ Define class labels in the same order as training
tumor_types = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Gemini API configuration
GEMINI_API_KEY = "AIzaSyCaq5T62OlV1gefNWmMsR1x4YATVyI4mRI"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# ✅ Preprocess input image
def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image.astype(np.float32)

# ✅ Gemini response generator
def get_gemini_response(prompt: str):
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if candidates:
            content_parts = candidates[0].get("content", {}).get("parts", [])
            return " ".join(part.get("text", "") for part in content_parts) if content_parts else "No response available."
        return "No candidates in response."
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return "AI response unavailable at the moment."

# ✅ Tumor info endpoint
@app.get("/tumor-info/{tumor_type}")
async def tumor_info(tumor_type: str):
    try:
        info_prompt = f"Give me detailed medical information about {tumor_type} brain tumor."
        tumor_info = get_gemini_response(info_prompt)
        return {"info": tumor_info}
    except Exception as e:
        print(f"❌ Tumor Info Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tumor information.")

# ✅ Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Step 1: Read and preprocess image
        contents = await file.read()
        image = preprocess_image(io.BytesIO(contents))

        # Step 2: Get raw prediction from the model
        raw_preds = model.predict(image).flatten()

        # Step 3: Get predicted class and confidence
        predicted_index = int(np.argmax(raw_preds))  # Use raw predictions directly
        confidence = float(np.max(raw_preds)) * 100  # Confidence based on raw prediction

        # Step 4: Handle low confidence predictions
        if confidence < 60:  # Adjusted threshold for better prediction confidence
            predicted_label = "Uncertain"
            confidence = 0  # Set confidence to 0 for uncertain predictions
        else:
            predicted_label = tumor_types[predicted_index]

        # Step 5: Handle Gemini API for tumor-specific report
        try:
            report_prompt = f"Generate a medical report with only highlighted points for a brain tumor diagnosis of type: {predicted_label}."
            ai_report = get_gemini_response(report_prompt)
        except Exception as api_error:
            print(f"❌ Gemini API error: {api_error}")
            ai_report = "AI-generated report is unavailable at the moment."

        return {
            "prediction": predicted_label,
            "confidence": round(confidence, 2),
            "ai_report": ai_report
        }

    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return {"detail": f"Error: {str(e)}"}, 500


# ✅ Chatbot endpoint
@app.post("/chatbot")
async def chatbot(data: dict):
    try:
        query = data.get("query")
        if not query:
            raise ValueError("Query cannot be empty.")
        response = get_gemini_response(f"Medical chatbot: {query}")
        return {"response": response}
    except Exception as e:
        print(f"❌ Chatbot Error: {e}")
        raise HTTPException(status_code=500, detail="Chatbot is currently unavailable.")
