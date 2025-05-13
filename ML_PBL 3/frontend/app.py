import streamlit as st
import requests

API_URL = "http://13.235.19.43:8000"  # Update if running on a different host

st.title("🧠 Brain Tumor Prediction & AI Medical Assistant")

# ✅ Upload image
uploaded_file = st.file_uploader("Upload a brain MRI image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

# ✅ Prediction section
if st.button("Predict Tumor"):
    if uploaded_file:
        try:
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(f"{API_URL}/predict", files=files)

            if response.headers.get("content-type") == "application/json":
                data = response.json()
                result = data.get("prediction", "Unknown")
                confidence = data.get("confidence", 0.0)
                ai_report = data.get("ai_report", "No AI Report available.")
                
                st.success(f"**Brain Tumor Prediction: {result}**")
                st.info(f"**Confidence Score:** {confidence:.2f}%")
                st.write("### AI-Generated Tumor Report 📝")
                st.write(ai_report)
            else:
                st.error("❌ Invalid or empty response from backend.")
                st.text(f"Raw response: {response.text}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("Please upload an image first.")

# ✅ Chatbot section
st.subheader("💬 AI Medical Chatbot")
user_query = st.text_input("Ask a medical question about brain tumors:")
if st.button("Ask AI"):
    if user_query.strip():
        try:
            response = requests.post(f"{API_URL}/chatbot", json={"query": user_query})
            if response.headers.get("content-type") == "application/json":
                st.write("**AI Response:**", response.json()["response"])
            else:
                st.error("❌ Invalid chatbot response.")
                st.text(response.text)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ✅ Tumor info
st.subheader("📚 Tumor Information Search")
tumor_type = st.text_input("Enter Tumor Type (e.g., Glioma, Meningioma, etc.)")
if st.button("Get Tumor Info"):
    if tumor_type.strip():
        try:
            response = requests.get(f"{API_URL}/tumor-info/{tumor_type}")
            if response.headers.get("content-type") == "application/json":
                st.write("### Tumor Information:")
                st.write(response.json()["info"])
            else:
                st.error("❌ Invalid info response.")
                st.text(response.text)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
