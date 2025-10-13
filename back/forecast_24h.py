import requests
import os
import json
import datetime
from dotenv import load_dotenv

import pandas as pd
import joblib
import torch
import numpy as np

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

def get_24h_forecast(data):
    current_time = datetime.datetime.now()
    max_time = current_time + datetime.timedelta(hours=24)

    result = []

    for item in data['list']:
        dt = datetime.datetime.fromtimestamp(item['dt'])
        
        if dt <= max_time:
            result.append({
                'date': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'temperature': item['main']['temp'],
                # 'humidity': item['main']['humidity'],
                # 'pressure': item['main']['pressure'],
                # 'wind_speed': item['wind']['speed']
            })

    return result

def extract_dates_temps(data):
    dates_24h = []
    temps_24h = []

    for item in data:
        dates_24h.append(item['date'])
        temps_24h.append(item['temperature'])

    display_times = [date.split(' ')[1][:5] for date in dates_24h]
    # temps_24h = [round(temp) for temp in temps_24h]

    return dates_24h, display_times, temps_24h

def fill_data_gaps(dates_24h, display_times, temps_24h):
    df = pd.DataFrame({'full_date': dates_24h, 'display_time': display_times, 'temperature': temps_24h})
    
    df['datetime'] = pd.to_datetime(df['full_date'])
    df = df.set_index('datetime')
    
    # заполняем промежутки во времени
    start_time = df.index.min()
    end_time = start_time + pd.Timedelta(hours=23)
    full_range = pd.date_range(start=start_time, end=end_time, freq='1h')
    
    # ресемплируем с интерполяцией
    df_resampled = df.reindex(full_range)
    df_resampled['temperature'] = df_resampled['temperature'].interpolate(method='linear')
    df_resampled['temperature'] = df_resampled['temperature'].fillna(method='ffill').fillna(method='bfill')
    
    full_dates = [t.strftime('%H:%M') for t in df_resampled.index]
    full_temps = [round(temp, 2) for temp in df_resampled['temperature']]
    
    return full_dates, full_temps

if __name__ == "__main__":
    data = get_data("london")
    forecast_24h = get_24h_forecast(data)
    # df = pd.DataFrame(forecast_24h)
    dates_24h, display_times, temps_24h = extract_dates_temps(forecast_24h)
    full_dates, full_temps = fill_data_gaps(dates_24h, display_times, temps_24h)
    print(full_dates, full_temps)