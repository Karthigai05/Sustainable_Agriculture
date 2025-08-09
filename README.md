

﻿# Fertilizer-Recommendation-System



# 🌱 AgriBot: Sustainable Agriculture Assistant with Voice & AI

AgriBot is a smart, voice-enabled chatbot system that assists farmers by recommending **organic fertilizers**, detecting **plant diseases from leaf images**, and providing **video/image suggestions** for eco-friendly farming practices. It is designed to support **sustainable agriculture** by leveraging AI models like **Random Forest** and **CNN**, making intelligent decisions based on real-time agricultural inputs.

![Screenshot (100)](https://github.com/user-attachments/assets/a95e70f8-4f86-4e23-9233-bb0fc36bdeba)
9f4)
![Screenshot (165)](https://github.com/user-attachments/assets/d5448995-7383-4228-9016-c55da568befc)
![Screenshot (101)](https://github.com/user-attachments/assets/d9208b27-0775-4408-81e7-dfe6ac41ff4c)
![Screenshot (151)](https://github.com/user-attachments/assets/0d9c7abd-8326-484f-846f-0af634bc7fc0)
![Screenshot (148)](https://github.com/user-attachments/assets/a5209077-10d4-4f8c-b7d9-9ef175347b40)
![Screenshot (149)](https://github.com/user-attachments/assets/0776feaa-3d38-4b13-bfa3-e4f7806645c5)
![Screenshot (110)](https://github.com/user-attachments/assets/f0b07129-2475-4d57-b3ba-d3a5d2c16318)




## 🚀 Features

- 🎤 **Voice-Assisted Input** using Speech Recognition & Text-to-Speech (TTS)
- 🌿 **Organic Fertilizer Recommendation** using Random Forest
- 🦠 **Plant Disease Detection** from leaf images using CNN
- 📊 **Input Parameters:** Temperature, Humidity, Moisture, Soil Type, Crop Type, N, P, K
- 🎥 **Video and Image Recommendation** for visual guidance
- 🗨️ **Chat History Display** for tracking inputs & outputs
- 📱 User-friendly **Streamlit Web App** interface

---

🧠## Machine Learning Models Used

| Module                    | Algorithm           | Model File                          |
|--------------------------|---------------------|-------------------------------------|
| Fertilizer Recommendation| Random Forest       | `fertilizer_model.pkl`              |
| Plant Disease Detection  | Convolutional Neural Network (CNN) | `plant_disease_prediction_model.h5` |
| Video Recommendation     | Random Forest       | `video_model.pkl`                   |
| Image Recommendation     | Random Forest       | `image_model.pkl`                   |





## 🛠️ Installation & Setup

1. Clone this repo
   ```bash
   git clone https://github.com/yourusername/AgriBot.git
   cd AgriBot
````

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app

   ```bash
   streamlit run main_app.py
   ```

---

## 🎤 Voice Control Instructions

* Click on the 🎤 Speak buttons to give voice input for numerical values.
* The system parses spoken numbers (e.g., "fifty four") into actual values.
* TTS provides voice feedback for clarity and guidance.

---

## 🧪 Inputs Required

* Temperature (°C)
* Humidity (%)
* Moisture (%)
* Soil Type– Selectable from list
* Crop Type – Selectable from list
* N (Nitrogen), P (Phosphorus), K (Potassium)

---

## 📊 Output Predictions

* ✅ Recommended Organic Fertilizer Name
* 🎬 Video link for how-to tutorials
* 🖼️ Image of recommended fertilizer or method
* 📈 Chat history of all interactions for traceability

---

## 🔧 Technologies Used

* Python, Streamlit
* TensorFlow / Keras – CNN model
* Scikit-learn / Joblib – Random Forest & preprocessing
* SpeechRecognition, pyttsx3– Voice input/output
* NumPy, PIL, wavio, sounddevice – Audio & image handling

---

## ♻️ Sustainable Agriculture Focus

* Promotes organic alternatives over chemical fertilizers
* Empowers farmers with real-time, localized solutions
* Supports visual learning and voice interaction for inclusivity

---

## 📈 Future Enhancements

* 🌐 Integrate IoT sensor data for automatic updates
* 📡 GPS-based location recommendations
* 📱 Deploy as a mobile app for broader access
* 🧑‍🌾 Add multi-language voice support for rural regions

---

## 👩‍💻 Contributors

* 👤 Karthigai Lakshmi C – Final Year CSE Student, RV College of Engineering
  Project Lead 



