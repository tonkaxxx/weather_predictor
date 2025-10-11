from flask import Flask, request, jsonify, send_from_directory
import requests
import os
import json
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

app = Flask(__name__)

def get_season(date: datetime) -> str:
    month = date.month
    if 3 <= month <= 5:
        return 'Весна'
    elif 6 <= month <= 8:
        return 'Лето'
    elif 9 <= month <= 11:
        return 'Осень'
    else:
        return 'Зима'

def get_data(city: str) -> dict:
    load_dotenv()
    api_key = os.getenv("API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"bad conn - {response.status_code}")
        return {}

def from_data_to_dataframe(data: dict) -> pd.DataFrame:
    if not data or 'list' not in data:
        return pd.DataFrame(columns=['season','date','temperature','humidity','pressure','wind_speed'])
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
            'date': dt_local.strftime('%Y-%m-%d %H:%M'),
            'temperature': main.get('temp'),
            'humidity': main.get('humidity'),
            'pressure': main.get('pressure'),
            'wind_speed': wind.get('speed')
        })
    df = pd.DataFrame(rows)
    for c in ['temperature', 'humidity', 'pressure', 'wind_speed']:
        if c in df.columns:
            df[c] = df[c].round(1)
    return df

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/get_weather', methods=['POST'])
def get_weather():
    data = request.get_json()
    city = data.get('city', '')
    raw_data = get_data(city)
    df = from_data_to_dataframe(raw_data)
    dates = df['date'].tolist()
    temperatures = df['temperature'].tolist()
    return jsonify({'dates': dates, 'temperatures': temperatures})

if __name__ == '__main__':
    app.run(debug=True)