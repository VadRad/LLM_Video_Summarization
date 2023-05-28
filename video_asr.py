import whisper

class TextExtractor:
    def __init__(self, model_path):
        self.model = whisper.load_model(model_path)
    
    def extract_text_from_audio(self, audio_path):
        result = self.model.transcribe(audio_path)
        return result["text"]
