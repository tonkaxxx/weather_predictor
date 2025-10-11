import requests
import os
import json
import pandas as pd
import joblib
import torch
import numpy as np

from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

def get_season(date: datetime) -> str:
    month = date.month

    if 3 <= month <= 5: return 'Весна'
    elif 6 <= month <= 8: return 'Лето'
    elif 9 <= month <= 11: return 'Осень'
    else: return 'Зима'

def get_data(city: str) -> dict:
    load_dotenv()
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
        day = dt_local.day

        main = item.get('main') or {}
        wind = item.get('wind') or {}

        rows.append({
            # 'season': get_season(dt_local),
            'date': day,
            'temperature': main.get('temp'),
            'humidity': main.get('humidity'),
            'pressure': main.get('pressure'),
            'wind_speed': wind.get('speed')
        })

    dataframe = pd.DataFrame(rows)

    daily_avg = dataframe.groupby('date').agg({
        # 'season': 'first',
        'temperature': 'mean',
        'humidity': 'mean',
        'pressure': 'mean',
        'wind_speed': 'mean'
    }).round(1)
    
    daily_avg = daily_avg.sort_index().head(5)
    print(daily_avg)
    return daily_avg

def from_df_to_nlist(df: pd.DataFrame) -> list[list[float]]:
    if df is None or df.empty:
        return []
    # превращаем нлисит
    result = df.to_numpy()
    if len(result) >= 5:
        last_5_days = result[-5:]
    return last_5_days
    
# def from_nlist_to_df(nlist):

if __name__ == "__main__":
    from predictor import predict_weather, load_trained_model # вопрос с импортом
    model = load_trained_model('weather_model.pth')
    scaler_x = joblib.load('scaler_x.pkl')
    scaler_y = joblib.load('scaler_y.pkl')

    raw_data = get_data("moscow")
    df = from_data_to_dataframe(raw_data)
    last_5_days = from_df_to_nlist(df)

    prediction = predict_weather(model, last_5_days, scaler_x, scaler_y)

    parameters = ['температура', 'влажность', 'давление', 'скорость ветра']
    units = ['°C', '%', 'гПа', 'м/с']

    print("\nпредсказания на след 3 дня:")
    for i in range(3):
        print(f"день +{i+1}: " + ", ".join([f"{param}: {prediction[i][j]:.1f}{unit}" 
        for j, (param, unit) in enumerate(zip(parameters, units))]))
