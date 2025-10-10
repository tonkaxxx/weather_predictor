import requests
import os
import json
import pandas as pd
import joblib

from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from flask import Flask, request, render_template

from back.back import get_season, get_data, get_daily_averages, from_data_to_dataframe
from back.predictor import predict_weather, load_trained_model

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
    model = load_trained_model('weather_model.pth')
    scaler_x = joblib.load('scaler_x.pkl')
    scaler_y = joblib.load('scaler_y.pkl')

    app.run(debug=True)

    

