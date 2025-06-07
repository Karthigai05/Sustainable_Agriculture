# FULLY UPDATED AGRIBOT STREAMLIT APP WITH ENHANCED UI & VOICE FEEDBACK
import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
import joblib
import pyttsx3
import re
import sounddevice as sd
from gtts import gTTS
import pygame
import time
import tempfile
import speech_recognition as sr
from PIL import Image
import wavio 

# ==== PATH SETUP ====
working_dir = os.path.dirname(os.path.abspath(__file__))

paths = {
    "disease_model": os.path.join(working_dir, "TrainedModel", "plant_disease_prediction_model.h5"),
    "class_indices": os.path.join(working_dir, "class_indices.json"),
    "fertilizer_model": os.path.join(working_dir, "fertilizer_model.pkl"),
    "video_model": os.path.join(working_dir, "video_model.pkl"),
    "image_model": os.path.join(working_dir, "image_model.pkl"),
    "scaler": os.path.join(working_dir, "standard_scaler.pkl"),
    "label_fertilizer": os.path.join(working_dir, "label_encoder_fertilizer.pkl"),
    "label_video": os.path.join(working_dir, "label_encoder_video.pkl"),
    "label_image": os.path.join(working_dir, "label_encoder_image.pkl"),
    "label_soil": os.path.join(working_dir, "label_encoder_soil.pkl"),
    "label_crop": os.path.join(working_dir, "label_encoder_crop.pkl"),
}

# ==== LOAD MODELS ====
@st.cache_resource
def load_models():
    return {
        "disease_model": tf.keras.models.load_model(paths["disease_model"]),
        "class_indices": json.load(open(paths["class_indices"])),
        "model_fertilizer": joblib.load(paths["fertilizer_model"]),
        "model_video": joblib.load(paths["video_model"]),
        "model_image": joblib.load(paths["image_model"]),
        "scaler": joblib.load(paths["scaler"]),
        "label_fertilizer": joblib.load(paths["label_fertilizer"]),
        "label_video": joblib.load(paths["label_video"]),
        "label_image": joblib.load(paths["label_image"]),
        "label_soil": joblib.load(paths["label_soil"]),
        "label_crop": joblib.load(paths["label_crop"]),
    }

models = load_models()

def loud(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def safe_delete(path, retries=5, delay=0.2):
    for _ in range(retries):
        try:
            os.remove(path)
            break
        except PermissionError:
            time.sleep(delay)

def speak(text_en, text_ta):
    pygame.mixer.init()

    # ===== English =====
    fd_en, path_en = tempfile.mkstemp(suffix=".mp3")
    os.close(fd_en)
    tts_en = gTTS(text=text_en, lang='en')
    tts_en.save(path_en)
    pygame.mixer.music.load(path_en)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.music.stop()
    time.sleep(0.3)  # Let OS release file handle
    safe_delete(path_en)

    # ===== Tamil =====
    fd_ta, path_ta = tempfile.mkstemp(suffix=".mp3")
    os.close(fd_ta)
    tts_ta = gTTS(text=text_ta, lang='ta')
    tts_ta.save(path_ta)
    pygame.mixer.music.load(path_ta)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.music.stop()
    time.sleep(0.3)
    safe_delete(path_ta)

# ==== PARSE SPOKEN NUMBERS ====
def parse_number_from_text(text):
    words_to_numbers = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
        "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100
    }
    try:
        match = re.findall(r'\d+', text)
        if match:
            return int(match[0])
        words = text.lower().split()
        number = 0
        temp = 0
        for word in words:
            if word in words_to_numbers:
                if word == "hundred":
                    temp *= 100
                else:
                    temp += words_to_numbers[word]
        number += temp
        return number
    except:
        return None

def record_and_recognize(prompt="Speak now"):
    st.info(prompt)
    fs = 44100
    duration = 5
    st.write("Recording for 5 seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    # Save recording as 16-bit PCM WAV using wavio
    wavio.write("input.wav", recording, fs, sampwidth=2)

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile("input.wav") as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None
    except ValueError:
        return None

# English to Tamil dictionary for class names
tamil_translations = {
    "Apple___Apple_scab": "ஆப்பிள் கறையீடு",
    "Apple___Black_rot": "ஆப்பிள் கருந்துண்டல்",
    "Apple___Cedar_apple_rust": "ஆப்பிள்___சீடர் ஆப்பிள் தாது வஞ்சகம்",
    "Apple___healthy": "ஆப்பிள்___நலமுடன்",
    "Blueberry___healthy": "ப்ளூபெர்ரி___நலமுடன்",
    "Cherry_(including_sour)___Powdery_mildew": "செர்ரி (புளிக்கும் சேர்த்து)___பொடி பூச்சி",
    "Cherry_(including_sour)___healthy": "செர்ரி___நலமுடன்",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "சோளம்___செர்கோஸ்போரா இலை புள்ளி",
    "Corn_(maize)___Common_rust_": "சோளம்___பொதுவான தாது வஞ்சகம்",
    "Corn_(maize)___Northern_Leaf_Blight": "சோளம்___வடக்கு இலை வாடல்",
    "Corn_(maize)___healthy": "சோளம்___நலமுடன்",
    "Grape___Black_rot": "திராட்சை___கருந்துண்டல்",
    "Grape___Esca_(Black_Measles)": "திராட்சை___எஸ்கா (கருப்புக் காச்சல்)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "திராட்சை___இலை வாடல் (இசாரியோப்சிஸ்)",
    "Grape___healthy": "திராட்சை___நலமுடன்",
    "Orange___Haunglongbing_(Citrus_greening)": "ஆரஞ்சு___ஹவுன்லொங்பிங் (சிட்ரஸ் பச்சை நோய்)",
    "Peach___Bacterial_spot": "பீச்___பாக்டீரியா புள்ளி",
    "Peach___healthy": "பீச்___நலமுடன்",
    "Pepper,_bell___Bacterial_spot": "குடைமிளகாய்___பாக்டீரியா புள்ளி",
    "Pepper,_bell___healthy": "குடைமிளகாய்___நலமுடன்",
    "Potato___Early_blight": "உருளைக்கிழங்கு___ஆரம்ப வாடல்",
    "Potato___Late_blight": "உருளைக்கிழங்கு___தாமத வாடல்",
    "Potato___healthy": "உருளைக்கிழங்கு___நலமுடன்",
    "Raspberry___healthy": "ராஸ்பெர்ரி___நலமுடன்",
    "Soybean___healthy": "சோயாபீன்___நலமுடன்",
    "Squash___Powdery_mildew": "ஸ்குவாஷ்___பொடி பூச்சி",
    "Strawberry___Leaf_scorch": "ஸ்ட்ராபெர்ரி___இலை எரிவு",
    "Strawberry___healthy": "ஸ்ட்ராபெர்ரி___நலமுடன்",
    "Tomato___Bacterial_spot": "தக்காளி___பாக்டீரியா புள்ளி",
    "Tomato___Early_blight": "தக்காளி___ஆரம்ப வாடல்",
    "Tomato___Late_blight": "தக்காளி___தாமத வாடல்",
    "Tomato___Leaf_Mold": "தக்காளி___இலை பூஞ்சை",
    "Tomato___Septoria_leaf_spot": "தக்காளி___செப்டோரியா இலை புள்ளி",
    "Tomato___Spider_mites Two-spotted_spider_mite": "தக்காளி___இரட்டைப்பட பிணைய பூச்சி",
    "Tomato___Target_Spot": "தக்காளி___இலக்கு புள்ளி",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "தக்காளி___மஞ்சள் இலை சுருக்க வைரஸ்",
    "Tomato___Tomato_mosaic_virus": "தக்காளி___தக்காளி மோசைக் வைரஸ்",
    "Tomato___healthy": "தக்காளி___நலமுடன்"
}

# ==== IMAGE PROCESSING ====
def load_and_preprocess_image(image_file, target_size=(224, 224)):
    img = Image.open(image_file).convert('RGB')
    img = img.resize(target_size)
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    return img_array

def predict_disease(image_file):
    img = load_and_preprocess_image(image_file)
    preds = models["disease_model"].predict(img)
    top_idx = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds) * 100
    label = models["class_indices"][str(top_idx)]
    tamil_label = tamil_translations.get(label, "மாற்றம் இல்லை")
    return label, tamil_label, confidence

# ==== VOICE NUMBER INPUT FIELD ====


def voice_number_input(key, label, min_val, max_val, default):
    # Separate internal value key to avoid conflict with st.number_input
    internal_key = f"val_{key}"

    if internal_key not in st.session_state:
        st.session_state[internal_key] = default

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("➖", key=f"dec_{key}"):
            st.session_state[internal_key] = max(min_val, st.session_state[internal_key] - 1)

    with col3:
        if st.button("➕", key=f"inc_{key}"):
            st.session_state[internal_key] = min(max_val, st.session_state[internal_key] + 1)

    with col2:
        # Show the number input but don’t bind it to session state directly
        val = st.number_input(label, min_value=min_val, max_value=max_val, value=st.session_state[internal_key], key=f"num_{key}")
        if val != st.session_state[internal_key]:
            st.session_state[internal_key] = val

    # Voice input section
    if st.button(f"🎤 Speak {label}", key=f"voice_{key}"):
        text = record_and_recognize(f"Speak {label}")
        if text:
            number = parse_number_from_text(text)
            if number is not None:
                number = max(min_val, min(max_val, number))
                st.session_state[internal_key] = number
                st.success(f"Updated {label} to {number}")
                loud(f"You said {number} for {label}")
            else:
                loud("Sorry, I could not understand the number.")
        else:
            loud("Sorry, I didn't catch that. Try again.")

    return st.session_state[internal_key]



st.title("🌾 AgriBot: Smart Agriculture Assistant")
st.markdown("---")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tab1, tab2 = st.tabs(["🦠 Plant Disease Detection", "🌿 Fertilizer & Media Recommendations"])

# === TAB 1: DISEASE ===
with tab1:
    st.header("📸 Upload a Plant Image to Detect Disease")
    uploaded_image = st.file_uploader("Upload plant image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        resized_img = image.resize((150,150))
        st.image(resized_img)   
        if st.button("🔍 Predict Disease"):
            with st.spinner("Analyzing..."):
                label_en, label_ta, confidence = predict_disease(uploaded_image)

                result_en = f"🩺 Disease Detected: **{label_en}** ({confidence:.2f}%)"
                result_ta = f" 🩺கண்டறியப்பட்ட நோய்: **{label_ta}** ({confidence:.2f}%)"
                
                st.success(result_en)
                st.success(result_ta)

                speak( f"plant disease detected is {label_en}"  , f"கண்டறியப்பட்ட நோய் {label_ta}"  )

# === TAB 2: FERTILIZER ===
with tab2:
    st.header("🧪 Organic Fertilizer, Video & Image Recommendations")

    col1, col2, col3 = st.columns(3)

    with col1:
        temp = voice_number_input("temp", "Temperature (°C)", 10, 50, 29)
        soil = st.selectbox("🌱 Soil Type", models["label_soil"].classes_)
        crop = st.selectbox("🌾 Crop Type", models["label_crop"].classes_)

    with col2:
        hum = voice_number_input("hum", "Humidity (%)", 10, 100, 54)
        n = voice_number_input("n", "Nitrogen (N)", 0, 140, 12)

    with col3:
        moisture = voice_number_input("moist", "Moisture (%)", 0, 100, 45)
        p = voice_number_input("p", "Phosphorus (P)", 0, 140, 36)
        k = voice_number_input("k", "Potassium (K)", 0, 140, 0)

    if st.button("📥 Get Recommendations"):
        with st.spinner("Processing..."):
            try:
                input_arr = np.array([[temp, hum, moisture,
                                       models["label_soil"].transform([soil])[0],
                                       models["label_crop"].transform([crop])[0],
                                       n, k, p]])
                scaled = models["scaler"].transform(input_arr)

                fert= models["model_fertilizer"].predict(scaled) 
                video= models ["model_video"].predict(scaled) 
                image= models["model_image"].predict(scaled) 
                fert_result = models["label_fertilizer"].inverse_transform(fert)[0] 
                video_result =models["label_video"].inverse_transform(video)[0] 
                image_result =models ["label_image"].inverse_transform(image)[0] 

                st.success(f"🌿 Recommended Fertilizer: **{fert_result}**")
                st.markdown(f"📊 **Inputs:** Temp: {temp}, Humidity: {hum}, Moisture: {moisture}, Soil: {soil}, Crop: {crop}, N: {n}, P: {p}, K: {k}")

                st.markdown("### 📺 Recommended Video:")
                st.video(video_result)
                st.markdown(f"[Open Video]({video_result})")

                
                st.image({image_result})
                                # Save to chat history
                user_input = f"User Inputs -> Temp: {temp}, Humidity: {hum}, Moisture: {moisture}, Soil: {soil}, Crop: {crop}, N: {n}, P: {p}, K: {k}"
                bot_response = f"Fertilizer: {fert_result}, Video: {video_result}, Image: {image_result}"
                st.session_state.chat_history.append((user_input, bot_response))

                # Display Chat History
                st.markdown("### 🗨️ Chat History")
                for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history[::-1], 1):
                    with st.expander(f"Chat #{len(st.session_state.chat_history) - i + 1}"):
                        st.markdown(f"👤 **User:** {user_msg}")
                        st.markdown(f"🤖 **AgriBot:** {bot_msg}")

                loud(f"Recommended fertilizer is {fert_result}")
                loud("Video and image are shown")


            except Exception as e:
                st.error("Error in recommendation process.")
                st.text(str(e))