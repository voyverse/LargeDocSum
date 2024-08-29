import dotenv
dotenv.load_dotenv()
import os
import requests

# Configuration
API_KEY = os.environ['AZURE_OPENAI_API_KEY']
ENDPOINT = os.environ['AZURE_OPENAI_ENDPOINT']
headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

payload = {
  "messages": [
    {
        "role": "user",
        "content": "What is the capital of France?"
    },
  ],
  "temperature": 0,
  "top_p": 0.95,
  "max_tokens": 800
}


try:
    response = requests.post(ENDPOINT, headers=headers, json=payload)
    response.raise_for_status() 
except requests.RequestException as e:
    raise SystemExit(f"Failed to make the request. Error: {e}")

print(response.json()['choices'][0]['message']['content'])