from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

url = "http://192.168.88.17:5000/api/get-24hrs"
model_name = "Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

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

    return content.strip()

@app.route('/api/ask-ai', methods=['POST'])
def api_ask_ai():
    try:
        data = request.get_json()
        city = data.get('city', '')
        user_input = data.get('user_input', '')

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