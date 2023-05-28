import openai

from config import MODEL_NAME, TEMPERATURE, MAX_TOKENS, API_KEY

class ChatModel:
    def __init__(self, model_name):
        self.model_name = model_name
        openai.api_key = API_KEY
    
    def make_api_call(self, messages):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
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
        ocr_message = self.model.create_user_message(f"The OCR output: {ocr_text}")
        asr_message = self.model.create_user_message(f"The ASR output: {asr_text}")
        prompt_message = self.model.create_user_message("Summarize the video based on the OCR and ASR.")
        
        conversation = [
            self.model.create_system_message("You are a helpful assistant."),
            ocr_message,
            asr_message
        ]
        
        response = self.model.make_api_call(conversation)
        return response