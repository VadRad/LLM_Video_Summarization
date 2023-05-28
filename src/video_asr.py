import whisper

class ASRHelper:
    def __init__(self, audio_path):
        self.model = whisper.load_model('base')
        self.audio_path = audio_path
    
    def extract_text_from_audio(self):
        result = self.model.transcribe(self.audio_path)
        return result["text"]
