from decouple import config

API_KEY = config('API_KEY')

# Constants
MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.8
MAX_TOKENS = 100