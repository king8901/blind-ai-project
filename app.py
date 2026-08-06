import streamlit as st
import torch
import cv2
import time
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import queue
import os

st.set_page_config(page_title="Live AI Blind Assistant", layout="centered")
st.title("👁️ Real-Time Audio Scene Navigation for Visually Impaired")
st.write("The camera runs continuously. Audio descriptions are generated automatically every 4 seconds.")

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model = YOLO('yolov8n.pt')
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return yolo_model, blip_processor, blip_model, device

yolo_model, blip_processor, blip_model, device = load_models()
frame_queue = queue.Queue(maxsize=1)

class LiveVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put(img)
        return frame

ctx = webrtc_streamer(
    key="live-nav", 
    video_processor_factory=LiveVideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:://google.com"]}]},
    media_stream_constraints={"video": True, "audio": False}
)

if ctx.state.playing:
    st.info("Continuous Real-Time Detection Active...")
    audio_placeholder = st.empty()
    text_placeholder = st.empty()
    
    if "last_processed" not in st.session_state:
        st.session_state.last_processed = time.time()

    while ctx.state.playing:
        if not frame_queue.empty():
            img_bgr = frame_queue.get()
            current_time = time.time()
            
            if current_time - st.session_state.last_processed > 4:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                
                cv2.imwrite("live_temp.jpg", img_bgr)
                yolo_results = yolo_model("live_temp.jpg", verbose=False)
                detected_objects = [yolo_model.names[int(box.cls)] for result in yolo_results for box in result.boxes]
                unique_objs = list(set(detected_objects))
                
                inputs = blip_processor(pil_img, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = blip_model.generate(**inputs, max_new_tokens=30)
                caption = blip_processor.decode(out, skip_special_tokens=True)
                
                if unique_objs:
                    final_text = f"The scene shows {caption}. Objects ahead are {', '.join(unique_objs)}."
                else:
                    final_text = f"The scene shows {caption}."
                
                text_placeholder.markdown(f"**Current View:** {final_text}")
                
                tts = gTTS(text=final_text, lang='en')
                tts.save("live_output.mp3")
                audio_placeholder.audio("live_output.mp3", format="audio/mp3", autoplay=True)
                
                st.session_state.last_processed = current_time
        time.sleep(0.1)
