from transformers import AutoModelForCausalLM, AutoTokenizer
import requests

model_name = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

url = "http://localhost:5000/api/get-24hrs"
response = requests.post(url, json={"city": "Москва"})

print(response.json())

weather_data = response.json()

print("Погодный ассистент запущен. Введите ваш вопрос:")

user_input = input("\nВы: ")

prompt = f"""
Ты - ассистент погоды. Отвечай ТОЛЬКО на основе данных о погоде.
Если данных нет - скажи "В прогнозе нет этой информации".

Данные о погоде на сегодня: {weather_data}

Вопрос: {user_input}

Ответ:"""

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    # enable_thinking=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

try:
    index = len(output_ids) - output_ids[::-1].index(151668)
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True)
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True)
except:
    content = tokenizer.decode(output_ids, skip_special_tokens=True)

print(f"Ассистент: {content.strip()}")
