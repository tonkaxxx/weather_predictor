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
                'humidity': item['main']['humidity'],
                'pressure': item['main']['pressure'],
                'wind_speed': item['wind']['speed']
            })

    return result

def extract_all_data(data):
    dates_24h = []
    temps_24h = []
    humidity_24h = []
    pressure_24h = []
    wind_speed_24h = []

    for item in data:
        dates_24h.append(item['date'])
        temps_24h.append(item['temperature'])
        humidity_24h.append(item['humidity'])
        pressure_24h.append(item['pressure'])
        wind_speed_24h.append(item['wind_speed'])

    display_times = [date.split(' ')[1][:5] for date in dates_24h]

    return dates_24h, display_times, temps_24h, humidity_24h, pressure_24h, wind_speed_24h

def fill_all_data_gaps(dates_24h, display_times, temps_24h, humidity_24h, pressure_24h, wind_speed_24h):
    df = pd.DataFrame({
        'full_date': dates_24h, 
        'display_time': display_times, 
        'temperature': temps_24h,
        'humidity': humidity_24h,
        'pressure': pressure_24h,
        'wind_speed': wind_speed_24h
    })
    
    df['datetime'] = pd.to_datetime(df['full_date'])
    df = df.set_index('datetime')
    
    # заполняем промежутки во времени
    start_time = df.index.min()
    end_time = start_time + pd.Timedelta(hours=23)
    full_range = pd.date_range(start=start_time, end=end_time, freq='1h')
    
    # ресемплируем с интерполяцией для всех параметров
    df_resampled = df.reindex(full_range)
    
    # интерполяция для разных типов данных
    df_resampled['temperature'] = df_resampled['temperature'].interpolate(method='linear')
    df_resampled['humidity'] = df_resampled['humidity'].interpolate(method='linear')
    df_resampled['pressure'] = df_resampled['pressure'].interpolate(method='linear')
    df_resampled['wind_speed'] = df_resampled['wind_speed'].interpolate(method='linear')
    
    # заполнение оставшихся пропусков
    df_resampled = df_resampled.fillna(method='ffill').fillna(method='bfill')
    
    full_dates = [t.strftime('%H:%M') for t in df_resampled.index]
    full_temps = [round(temp, 2) for temp in df_resampled['temperature']]
    full_humidity = [round(hum, 2) for hum in df_resampled['humidity']]
    full_pressure = [round(pres, 2) for pres in df_resampled['pressure']]
    full_wind_speed = [round(wind, 2) for wind in df_resampled['wind_speed']]
    
    return full_dates, full_temps, full_humidity, full_pressure, full_wind_speed

if __name__ == "__main__":
    data = get_data("london")
    forecast_24h = get_24h_forecast(data)
    dates_24h, display_times, temps_24h, humidity_24h, pressure_24h, wind_speed_24h = extract_all_data(forecast_24h)
    full_dates, full_temps, full_humidity, full_pressure, full_wind_speed = fill_all_data_gaps(
        dates_24h, display_times, temps_24h, humidity_24h, pressure_24h, wind_speed_24h
    )
    
    print("Времена:", full_dates)
    print("Температуры:", full_temps)
    print("Влажность:", full_humidity)
    print("Давление:", full_pressure)
    print("Скорость ветра:", full_wind_speed)