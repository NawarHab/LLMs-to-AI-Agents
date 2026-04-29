import requests

API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"
# replace the random example string with your own key, take care not to share this file with others - otherwise the can use your ressoures
headers = {"Authorization": ""}
payload = {
    "messages": [
        {
            "role": "user",
            "content": "Which number yields the same result when it is added or multiplied with itself?",
        }
    ],
    "model": "deepseek/deepseek-v3-0324",
}

response = requests.post(API_URL, headers=headers, json=payload)
# print(response.json()["choices"][0]["message"])

from IPython.display import Markdown, display

display(Markdown(response.json()["choices"][0]["message"]["content"]))
