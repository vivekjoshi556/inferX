import os
from dotenv import load_dotenv
from inferX import load_model

load_dotenv()

model = load_model(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    auth_token=os.getenv('HF_TOKEN')
)