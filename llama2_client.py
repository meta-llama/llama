import requests

url = "http://10.60.57.21:9001/generate"  # Replace with the actual URL of your API endpoint

data = {
    "prompts": [
        "李白是位诗人"
        "李白的作品有哪些",
    ],
    "max_gen_len": 64,
    "temperature": 0.6,
    "top_p": 0.9
}

response = requests.post(url, json=data)
if response.status_code == 200:
    generated_texts = response.json()["generated_texts"]
    for generated_text in generated_texts:
        print(generated_text["prompt"])
        print("> " + generated_text["generation"])
        print("\n==================================\n")
else:
    print("Error:", response.status_code, response.text)
