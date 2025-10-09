import requests
import os
import json
import pandas as pd

from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from flask import Flask, request, render_template

def get_season(date: datetime) -> str:
    month = date.month

    if 3 <= month <= 5: return 'Весна'
    elif 6 <= month <= 8: return 'Лето'
    elif 9 <= month <= 11: return 'Осень'
    else: return 'Зима'

def get_data(city: str) -> dict:
    env_path = os.path.join(os.path.dirname(__file__), 'back', '.env')
    load_dotenv(env_path)
    api_key = os.getenv("API_KEY")

    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)

    # проверка подключения
    if response.status_code == 200:
        json_data = json.dumps(response.json(), indent=4, ensure_ascii=False)
        print("conn est")
    else:
        print(f"bad conn - {response.status_code}")
        return {}

    # переделываем данные в дикт
    data = json.loads(json_data)
    return data


def from_data_to_dataframe(data: dict) -> pd.DataFrame:
    if not data or 'list' not in data:
        return pd.DataFrame(columns=['season','date','temperature','humidity','pressure','wind_speed'])

    # смящение часового пояса
    timezone_offset = data.get('city', {}).get('timezone', 0)

    rows = []
    for item in data['list']:
        timestamp = item.get('dt')
        if timestamp is None:
            continue

        dt_local = datetime.fromtimestamp(timestamp, timezone.utc) + timedelta(seconds=timezone_offset)

        main = item.get('main') or {}
        wind = item.get('wind') or {}

        rows.append({
            'season': get_season(dt_local),
            'date': dt_local,
            'temperature': main.get('temp'),
            'humidity': main.get('humidity'),
            'pressure': main.get('pressure'),
            'wind_speed': wind.get('speed')
        })

    dataframe = pd.DataFrame(rows)

    # округление чисел
    numeric_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
    for c in numeric_cols:
        if c in dataframe.columns:
            dataframe[c] = dataframe[c].round(1)

    return dataframe

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-weather', methods=['POST'])
def get_weather():
    city = request.form.get('city')
    data = get_data(city)
    print(data)
    df = from_data_to_dataframe(data)

    if not df.empty:
        # Преобразуем DataFrame в красивую HTML таблицу
        html_table = df.to_html(classes='table table-striped', index=False, border=0)
        return f"""
        <html>
        <head>
            <title>Погода в {city}</title>
            <link href="/static/style_website.css" rel="stylesheet">
            <style>
                .table {{ width: 100%; border-collapse: collapse; }}
                .table th, .table td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                .table th {{ background-color: #f2f2f2; }}
                .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Погода в городе: {city}</h1>
                {html_table}
                <br>
                <a href="/">Назад к поиску</a>
            </div>
        </body>
        </html>
        """
    else:
        return f"""
        <html>
        <head>
            <link href="/static/style_website.css" rel="stylesheet">
        </head>
        <body>
            <div class="container">
                <h2>Ошибка</h2>
                <p>Не удалось получить данные для города: {city}</p>
                <a href="/">Назад к поиску</a>
            </div>
        </body>
        </html>
        """


if __name__ == "__main__":
    raw_data = get_data("moscow")
    df = from_data_to_dataframe(raw_data)
    # print(df)
    app.run(debug=True)

    

