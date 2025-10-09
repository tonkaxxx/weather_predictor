import requests
import os
import json
import pandas as pd

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

def get_daily_averages(df: pd.DataFrame) -> list[list[float]]:
    """
    Принимает DataFrame с колонками:
    ['season', 'date', 'temperature', 'humidity', 'pressure', 'wind_speed']

    Возвращает вложенный список вида:
    [[temperature_avrg, humidity_avrg, pressure_avrg, wind_speed_avrg], ...]
    по каждому дню.
    """
    if df is None or df.empty:
        return []

    # ставим дату как индекс
    df = df.set_index('date')

    # группируем по дням и считаем средние значения
    daily_averages = df.resample('D').agg({
        'temperature': 'mean',
        'humidity': 'mean',
        'pressure': 'mean',
        'wind_speed': 'mean'
    })

    # округляем значения
    daily_averages = daily_averages.round(1)

    # превращаем в список списков
    result = daily_averages.to_numpy().tolist()
    return result

if __name__ == "__main__":
    raw_data = get_data("moscow")
    df = from_data_to_dataframe(raw_data)
    res = get_daily_averages(df)
    print(res)