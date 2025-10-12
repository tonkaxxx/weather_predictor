import requests
import os
import json
import pandas as pd
import joblib

from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from flask import Flask, request, render_template

from back.back import get_season, get_data, from_df_to_nlist, from_data_to_dataframe
from back.predictor import predict_weather, load_trained_model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-weather', methods=['POST'])
def get_weather():
    # получаем данные для нейронки и для таблицы
    city = request.form.get('city')
    data = get_data(city)
    df = from_data_to_dataframe(data)
    last_5_days = from_df_to_nlist(df)
    
    if df.empty:
        return render_template(
            'error.html',
            city=city
        )

    prediction = predict_weather(model, last_5_days, scaler_x, scaler_y)

    # дата для таблицы
    last_date = df.index.max()
    new_dates = [last_date + 1, last_date + 2, last_date + 3]

    ai_df = pd.DataFrame(
        prediction, 
        columns=['temperature', 'humidity', 'pressure', 'wind_speed'],
        index=new_dates
    ).round(1)
    df = pd.concat([df, ai_df]) # соед 2 дфа
    df = df.round({
        'temperature': 0,
        'humidity': 0, 
        'pressure': 0,
        'wind_speed': 1
    }).astype({
        'temperature': int,
        'humidity': int,
        'pressure': int
    })
    df = df.rename(columns={
        'temperature': 'Температура',
        'humidity': 'Влажность', 
        'pressure': 'Давление',
        'wind_speed': 'Скорость ветра'
    })
        
    real_temperatures = [float(row[0]) for row in last_5_days]
    temperatures = [float(row[0]) for row in prediction]
    
    days = []
    first_date = int(df.index.min())
    for i in range(8):
        days.append(first_date + i)

    if not df.empty:
        html_table = df.to_html(classes='table table-striped', index=True, border=0)
        
        return render_template(
            'get-weather.html',
            city=city,
            html_table=html_table,
            temperatures=temperatures,
            real_temperatures=real_temperatures,
            days=days
        )

if __name__ == "__main__":
    model = load_trained_model('weather_model.pth')
    scaler_x = joblib.load('scaler_x.pkl')
    scaler_y = joblib.load('scaler_y.pkl')

    app.run(host='0.0.0.0', port=5000, debug=False)