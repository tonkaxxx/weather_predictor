# from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
# import torch
import uuid
import json
from dotenv import load_dotenv
import os

app = Flask(__name__)
CORS(app)

# url = "http://192.168.88.79:5000/api/get-24hrs"
# model_name = "Qwen/Qwen3-4B-Instruct-2507"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     dtype=torch.float16,
#     device_map="auto",
#     trust_remote_code=True,
#     cache_dir="./model_cache" 
# )


def get_token(auth_token, scope='GIGACHAT_API_PERS'):
    """
      Выполняет POST-запрос к эндпоинту, который выдает токен.

      Параметры:
      - auth_token (str): токен авторизации, необходимый для запроса.
      - область (str): область действия запроса API. По умолчанию — «GIGACHAT_API_PERS».

      Возвращает:
      - ответ API, где токен и срок его "годности".
      """
    # Создадим идентификатор UUID (36 знаков)
    rq_uid = str(uuid.uuid4())

    # API URL
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

    # Заголовки
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': rq_uid,
        'Authorization': f'Basic {auth_token}'
    }

    # Тело запроса
    payload = {
        'scope': scope
    }

    try:
        # Делаем POST запрос с отключенной SSL верификацией
        # (можно скачать сертификаты Минцифры, тогда отключать проверку не надо)
        response = requests.post(url, headers=headers, data=payload, verify=False)
        return response
    except requests.RequestException as e:
        print(f"Ошибка: {str(e)}")
        return -1

def get_chat_completion(auth_token, user_message):
    """
    Отправляет POST-запрос к API чата для получения ответа от модели GigaChat.

    Параметры:
    - auth_token (str): Токен для авторизации в API.
    - user_message (str): Сообщение от пользователя, для которого нужно получить ответ.

    Возвращает:
    - str: Ответ от API в виде текстовой строки.
    """
    # URL API, к которому мы обращаемся
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    # Подготовка данных запроса в формате JSON
    payload = json.dumps({
        "model": "GigaChat",  # Используемая модель
        "messages": [
            {
                "role": "user",  # Роль отправителя (пользователь)
                "content": user_message  # Содержание сообщения
            }
        ],
        "temperature": 1,  # Температура генерации
        "top_p": 0.1,  # Параметр top_p для контроля разнообразия ответов
        "n": 1,  # Количество возвращаемых ответов
        "stream": False,  # Потоковая ли передача ответов
        "max_tokens": 512,  # Максимальное количество токенов в ответе
        "repetition_penalty": 1,  # Штраф за повторения
        "update_interval": 0  # Интервал обновления (для потоковой передачи)
    })

    # Заголовки запроса
    headers = {
        'Content-Type': 'application/json',  # Тип содержимого - JSON
        'Accept': 'application/json',  # Принимаем ответ в формате JSON
        'Authorization': f'Bearer {auth_token}'  # Токен авторизации
    }

    # Выполнение POST-запроса и возвращение ответа
    try:
        response = requests.request("POST", url, headers=headers, data=payload, verify=False)
        return response
    except requests.RequestException as e:
        # Обработка исключения в случае ошибки запроса
        print(f"Произошла ошибка: {str(e)}")
        return -1


def ask_gigachat(user_input, city):
    load_dotenv()
    auth = os.getenv("AUTH")

    response_giga = get_token(auth)
    if response != 1:
        print(response_giga.text)
        giga_token = response_giga.json()['access_token']

    response = requests.post(url, json={"city": city})

    # нужно сделать обработку неудачи в апи
    weather_data = response.json()['for_ai']
    print(weather_data)

    prompt = f"""
    Ты - ассистент погоды. Отвечай ТОЛЬКО на основе данных о погоде. Не использу язык разметки markdown.
    Если данных нет - скажи "В прогнозе нет этой информации".

    Данные о погоде на сегодня: {weather_data}
    Если влажность больше 85 процентов - говори что идет дождь, а если температура минусовая, то говори, что идет снег.Но не проговаривай условие.

    Вопрос: {user_input}

    Ответ:"""

    answer = get_chat_completion(giga_token, prompt)
    answer.json()
    return answer.json()['choices'][0]['message']['content']

def ask_qwen(user_input, city):
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

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128
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
        if user_input == "test123":
            return jsonify({
                'status': 'success',
                'response': "На сегодня температура постепенно растёт с утра до дня, достигая максимума около 6,93 градусов в 12:00, затем немного снижается. Влажность стабильно повышается с утра до вечернего времени, достигая 77% к вечеру. Ветер умеренный, скорость варьируется от 5,95 до 8,16 м/с. Влажность не превышает 77%, поэтому дождя нет. На сегодня рекомендуется надеть теплую одежду — плед, свитер или кофту — из-за низкой температуры в вечернее время. В дневное время можно использовать легкую одежду, но при этом не забывать о тепле. Ветер может быть ощутимым, особенно в вечерние часы, поэтому можно взять пальто или куртку."
            }), 200
        
        data = request.get_json()
        city = data.get('city', '')
        user_input = data.get('user_input', '')
        if user_input == "какая сегодня погода?":
            return jsonify({
                'status': 'success',
                'response': "Температура днем от 4.7 градуса до -2.4 градуса ночью. Ночью и утром идет дождь. Давление от 1004 до 1017 гПа. Ветер от 2.0 до 5.4 м/с."
            }), 200
        
        data = request.get_json()
        city = data.get('city', '')
        user_input = data.get('user_input', '')
        if user_input == "что мне сегодня надеть?":
            return jsonify({
                'status': 'success',
                'response': "Сегодня температура в течение дня будет понижаться с 4.7 градуса до -2.4 градуса. Ночью и утром идет дождь. Рекомендую надеть теплую, непромокаемую одежду и обувь. Не забудьте шапку и перчатки к вечеру, так как ожидается мороз."
            }), 200

        response = ask_gigachat(user_input, city)
        
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

    print(ask_gigachat("какая погода на сегодня?", "Москва"))