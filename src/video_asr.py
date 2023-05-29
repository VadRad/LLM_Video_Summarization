import whisper
import tempfile
import streamlit as st

@st.cache_resource
def load_model():
    return whisper.load_model('base')


class ASRHelper:
    def __init__(self):
        self.model = load_model()
    
    def extract_text_from_audio(self, video_bytes):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as temp_video_file:
            temp_video_file.write(video_bytes)
            result = self.model.transcribe(temp_video_file.name)
        return result["text"]
