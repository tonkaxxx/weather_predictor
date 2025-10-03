import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

# для дебага
def get_season(date):
    month = date.month
    day = date.day
    
    if 3 <= month <= 5: return 'Весна'
    elif 6 <= month <= 8: return 'Лето'
    elif 9 <= month <= 11: return 'Осень'
    else: return 'Зима'

def generate_weather_data(days=365):
    # создаем массив дат 365 с дневным шагом 
    dates = pd.date_range(start='2025-01-01', periods=days, freq='D')
    # создаем ндлист от 1 до кол-ва дней  
    time = np.arange(days)
    
    # основная сезонная синусоида
    base_temperature = 7.5 + 22.5 * np.sin(2 * np.pi * time / 365 - np.pi/2)
    
    # генерируем случайные точки и интерполируем
    temperature_num_points = days // 30  # точки раз в 30 дней
    temperature_random_points = np.random.normal(0, 3, temperature_num_points) # создаем 12 точек от 0 до 3
    temperature_random_days = np.linspace(0, days-1, temperature_num_points) # расставляем их равномерно по году
    temperature_smooth_noise = np.interp(time, temperature_random_days, temperature_random_points) # связываем вершины и сглаживаем результат
    
    # генерируем массив из дейс и случайного числа для него 
    daily_random_noise = np.array([np.round(np.random.uniform(-1, 1), 1) for i in range(days)])
    
    # складываем синусоиду с шумом
    temperature = base_temperature + temperature_smooth_noise 
    
    # сглаживаем скользящеим среднеим по 3 дням для устранения резких скачков
    temp_series = pd.Series(temperature) # переводим в другой тип данных для фунrolling
    temperature = temp_series.rolling(window=3, center=True, min_periods=1).mean().values + daily_random_noise

    # базовая зависимость от температуры
    base_humidity = 75 - 0.48 * temperature
    
    # генерируем случайные точки и интерполируем
    humidity_num_points = days // 30
    humidity_random_points = np.random.normal(0, 6, humidity_num_points) # создаем 12 точек от 0 до 8
    humidity_random_days = np.linspace(0, days-1, humidity_num_points) # расставляем их равномерно по году
    humidity_smooth_noise = np.interp(time, humidity_random_days, humidity_random_points) # связываем вершины и сглаживаем результат
    
    humidity = base_humidity + humidity_smooth_noise + daily_random_noise * 2
    humidity = np.clip(humidity, 40, 95)
    
    # синусоида давления (тк зависит от времени года) + шум
    pressure = 1015 + 5 * np.sin(2 * np.pi * time / 365) + np.random.normal(0, 1.5, days)
    
    # синусоида ветра + шум
    base_wind = 2.5 + np.sin(2 * np.pi * time / 365) + np.random.normal(0, 1, days)
    wind_speed = np.clip(base_wind, 0.5, 5)
    
    # времена года для дебага
    seasons = [get_season(date) for date in dates]
    
    df = pd.DataFrame({
        'season': seasons,
        'date': dates,
        'temperature': np.round(temperature, 1),
        'humidity': np.round(humidity, 1),
        'pressure': np.round(pressure, 1),
        'wind_speed': np.round(wind_speed, 1)
    })
    
    return df

# вычислить за 365 и запринтовать
weather_df = generate_weather_data(365)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(weather_df)