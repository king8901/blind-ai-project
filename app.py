import streamlit as st
import torch
import time
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import os

st.set_page_config(page_title="Live AI Blind Assistant", layout="centered")
st.title("👁️ Zero-Interaction Audio Navigation for Visually Impaired")
st.write("The AI scans the scene and speaks continuously using your webcam.")

# ১. মডেল ক্যাশিং (CPU অপ্টিমাইজড)
@st.cache_resource
def load_models():
    yolo_model = YOLO('yolov8n.pt')
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return yolo_model, blip_processor, blip_model

yolo_model, blip_processor, blip_model = load_models()

# ২. লাইভ ক্যামেরা ইনপুট
img_file = st.camera_input("Turn on Camera for Continuous Scanning")

if img_file is not None:
    # ইমেজ লোড করা
    pil_img = Image.open(img_file).convert('RGB')
    
    with st.spinner("AI is analyzing the current view..."):
        # অস্থায়ী সেভ করা YOLO-র জন্য
        temp_path = "live_snapshot.jpg"
        pil_img.save(temp_path)
        
        # ক) অবজেক্ট ডিটেকশন (YOLOv8)
        yolo_results = yolo_model(temp_path, verbose=False)
        detected_objects = [yolo_model.names[int(box.cls)] for result in yolo_results for box in result.boxes]
        unique_objs = list(set(detected_objects))
        
        # খ) সিন ক্যাপশনিং (NLP Transformer)
        inputs = blip_processor(pil_img, return_tensors="pt")
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=30)
        caption = blip_processor.decode(out, skip_special_tokens=True)
        
        # গ) ইনফরমেশন ফিউশন
        if unique_objs:
            final_text = f"Ahead of you, {caption}. Detected items are {', '.join(unique_objs)}."
        else:
            final_text = f"Ahead of you, {caption}."
            
        # ঘ) ইউআই রেন্ডার এবং অডিও প্লেব্যাক
        st.markdown(f"📡 **AI Voice Guide:** {final_text}")
        
        tts = gTTS(text=final_text, lang='en')
        tts.save("feedback.mp3")
        st.audio("feedback.mp3", format="audio/mp3", autoplay=True)
        
        # ক্লিনআপ
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    # ৫ সেকেন্ড পর পেজটি অটো-রিফ্রেশ হবে যাতে ক্যামেরা পরবর্তী ফ্রেম নিজে থেকেই ক্যাপচার করে
    time.sleep(5)
    st.rerun()
