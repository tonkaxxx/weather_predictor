from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch

app = Flask(__name__)
CORS(app)

url = "http://192.168.88.17:5000/api/get-24hrs"
model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    cache_dir="./model_cache" 
)

# компилим модель для ускорения
if hasattr(torch, 'compile'):
    model = torch.compile(model)

def ask_ai(user_input, city):
    response = requests.post(url, json={"city": city})

    # нужно сделать обработку неудачи в апи
    weather_data = response.json()['for_ai']
    print(weather_data)

    # print("Погодный ассистент запущен. Введите ваш вопрос:")
    # user_input = input("\nВы: ")

    prompt = f"""
    Ты - ассистент погоды. Отвечай ТОЛЬКО на основе данных о погоде. Не использу язык разметки markdown.
    Если данных нет - скажи "В прогнозе нет этой информации".

    Данные о погоде на сегодня: {weather_data}
    Если влажность больше 85 процентов - говори что идет дождь. Но не проговаривай условие.

    Вопрос: {user_input}

    Ответ:"""

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            num_return_sequences=1,
            early_stopping=True
        )

    response_text = tokenizer.decode(
        generated_ids[0][len(model_inputs.input_ids[0]):], 
        skip_special_tokens=True
    ).strip()

    return response_text

@app.route('/api/ask-ai', methods=['POST'])
def api_ask_ai():
    try:
        data = request.get_json()
        city = data.get('city', '')
        user_input = data.get('user_input', '')
        if user_input == "test123":
            return jsonify({
                'status': 'success',
                'response': "На сегодня температура постепенно растёт с утра до дня, достигая максимума около 6,93 градусов в 12:00, затем немного снижается. Влажность стабильно повышается с утра до вечернего времени, достигая 77% к вечеру. Ветер умеренный, скорость варьируется от 5,95 до 8,16 м/с. Влажность не превышает 77%, поэтому дождя нет. На сегодня рекомендуется надеть теплую одежду — плед, свитер или кофту — из-за низкой температуры в вечернее время. В дневное время можно использовать легкую одежду, но при этом не забывать о тепле. Ветер может быть ощутимым, особенно в вечерние часы, поэтому можно взять пальто или куртку."
            }), 200

        response = ask_ai(user_input, city)
        
        return jsonify({
            'status': 'success',
            'response': response
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050, debug=False)

    print(ask_ai("какая погода на сегодня?", "Москва"))