import os
import requests
from dotenv import load_dotenv

load_dotenv()

url = "https://ml-64288d82-5dd.go01-dem.ylcu-atmi.cloudera.site/namespaces/serving-default/endpoints/goes---nemotron-49b-throughput-l40s/v1/chat/completions"
api_key = os.getenv("WINDSURF_LLM_API_KEY")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

data = {
    "model": "nvidia/llama-3.3-nemotron-super-49b-v1",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 50
}

try:
    response = requests.post(url, headers=headers, json=data)
    print(f"Status Code: {response.status_code}")
    print("Response:", response.text)
except Exception as e:
    print(f"Error: {str(e)}")