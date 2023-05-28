import openai
import streamlit as st

class ChatModel:
    def __init__(self, model_name):
        self.model_name = model_name
        openai.api_key = st.session_state.get("OPENAI_API_KEY")
    
    def make_api_call(self, messages):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages
        )
        return response['choices'][0]['message']['content']
    
    def create_user_message(self, content):
        return {"role": "user", "content": content}
    
    def create_assistant_message(self, content):
        return {"role": "assistant", "content": content}
    
    def create_system_message(self, content):
        return {"role": "system", "content": content}
    

class ChatHandler:
    def __init__(self, model_name):
        self.model = ChatModel(model_name)
    
    def process_ocr_and_asr(self, ocr_text, asr_text):
        system_message = self.model.create_system_message("You are a summarization application which creates a meaningfull summary for the provided video text.")
        ocr_message = self.model.create_user_message(f"The OCR output: {ocr_text}")
        asr_message = self.model.create_user_message(f"The ASR output: {asr_text}")
        prompt_message = self.model.create_user_message("Summarize the video based on the OCR and ASR. Both OCR and ASR might contain mistakes, but OCR data is preferable for spelling check")
        
        conversation = [
            system_message,
            ocr_message,
            asr_message,
            prompt_message
        ]
        
        response = self.model.make_api_call(conversation)
        return response