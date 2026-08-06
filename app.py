import streamlit as st
import torch
import time
from PIL import Image
from ultralytics import YOLO
from gtts import gTTS
import os

st.set_page_config(page_title="Live AI Blind Assistant", layout="centered")
st.title("👁️ Zero-Interaction Audio Navigation for Visually Impaired")
st.write("The AI scans the scene and speaks continuously using your webcam.")

# ১. লাইটওয়েট মডেল লোড করা (মাত্র ১৫ এমবি র‍্যাম নেবে)
@st.cache_resource
def load_models():
    yolo_model = YOLO('yolov8n.pt')  # Nano CV Model
    return yolo_model

yolo_model = load_models()

# ২. লাইভ ক্যামেরা ইনপুট
img_file = st.camera_input("Turn on Camera for Continuous Scanning")

if img_file is not None:
    pil_img = Image.open(img_file).convert('RGB')
    
    with st.spinner("AI is analyzing the current view..."):
        temp_path = "live_snapshot.jpg"
        pil_img.save(temp_path)
        
        # ক) রিয়েল-টাইম অবজেক্ট ট্র্যাকিং
        yolo_results = yolo_model(temp_path, verbose=False)
        detected_objects = [yolo_model.names[int(box.cls)] for result in yolo_results for box in result.boxes]
        unique_objs = list(set(detected_objects))
        
        # খ) লাইটওয়েট রুল-বেসড কনটেক্সট জেনারেশন (ফ্রি সার্ভারের জন্য পারফেক্ট)
        if unique_objs:
            if "person" in unique_objs:
                final_text = f"Be careful. A person and {', '.join([x for x in unique_objs if x != 'person'])} detected ahead of you."
            else:
                final_text = f"Navigation path has {', '.join(unique_objs)} ahead of you."
        else:
            final_text = "The path ahead appears clear. Proceed safely."
            
        # গ) ইউআই রেন্ডার এবং অটো-প্লেব্যাক
        st.subheader("📡 AI Voice Guide")
        st.success(final_text)
        
        tts = gTTS(text=final_text, lang='en')
        tts.save("feedback.mp3")
        st.audio("feedback.mp3", format="audio/mp3", autoplay=True)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    # পেজ রিফ্রেশ টাইমার
    time.sleep(4)
    st.rerun()
