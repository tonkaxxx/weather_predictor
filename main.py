import requests
import os
import json
import pandas as pd
import joblib

from dotenv import load_dotenv
import datetime
from flask import Flask, request, render_template

from back.back import get_season, get_data, from_df_to_nlist, from_data_to_dataframe
from back.predictor import predict_weather, load_trained_model
from back.forecast_24h import get_24h_forecast, extract_all_data, fill_all_data_gaps

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-8days', methods=['POST'])
def get_8days():
    # получаем данные для нейронки и для таблицы
    city = request.form.get('city')
    if city.isdigit(): # почему-то при 4ех или 5ти значных числах апи выдает города... 
        return render_template(
            'error.html',
            city=city
        )
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

    print(last_5_days)
    print(prediction)

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
        'temperature': 'Температура, °C',
        'humidity': 'Влажность, %', 
        'pressure': 'Давление, гПа',
        'wind_speed': 'Скорость ветра, м/с'
    })
        
    real_temperatures = [float(row[0]) for row in last_5_days]
    temperatures = [float(row[0]) for row in prediction]
    
    days = []
    first_date = int(df.index.min())
    for i in range(8):
        days.append(first_date + i)

    if not df.empty:
        days_ru = {
            'Monday': 'Пн',
            'Tuesday': 'Вт', 
            'Wednesday': 'Ср',
            'Thursday': 'Чт',
            'Friday': 'Пт',
            'Saturday': 'Сб',
            'Sunday': 'Вс'
        }
        
        current_date = datetime.datetime.now()
        new_index = []
        
        for i in range(len(df)):
            date = current_date + datetime.timedelta(days=i)
            day_name_ru = days_ru[date.strftime('%A')]
            day_number = date.day
            
            if i == 0:
                new_index.append("Сегодня")
            elif i == 1:
                new_index.append("Завтра")
            else:
                new_index.append(f"{day_name_ru}, {day_number:02d}")
        
        df.index = new_index

        html_table = df.to_html(classes='table table-striped', index=True, border=1)

        return render_template(
            'get-8days.html',
            city=city,
            html_table=html_table,
            temperatures=temperatures,
            real_temperatures=real_temperatures,
            days=days,
        )

@app.route('/get-24hrs', methods=['POST'])
def get_24hrs():
    city = request.form.get('city')
    if city.isdigit(): # почему-то при 4ех или 5ти значных числах апи выдает города... 
        return render_template(
            'error.html',
            city=city
        )
    data = get_data(city)

    if not data:
        return render_template(
            'error.html',
            city=city
        )

    forecast_24h = get_24h_forecast(data)
    dates_24h, display_times, temps_24h, humidity_24h, pressure_24h, wind_speed_24h = extract_all_data(forecast_24h)
    full_dates, full_temps, full_humidity, full_pressure, full_wind_speed = fill_all_data_gaps(dates_24h, display_times, temps_24h, humidity_24h, pressure_24h, wind_speed_24h)

    time_series = pd.to_datetime(dates_24h).strftime('%H:%M')
    hours_mins = time_series.tolist()

    df24 = pd.DataFrame({
        'temperature': temps_24h,
        'humidity': humidity_24h,
        'pressure': pressure_24h,
        'wind_speed': wind_speed_24h
    }, index=hours_mins)
    df24 = df24.rename(columns={
        'temperature': 'Температура, °C',
        'humidity': 'Влажность, %', 
        'pressure': 'Давление, гПа',
        'wind_speed': 'Скорость ветра, м/с'
    })

    df = from_data_to_dataframe(data)
    last_5_days = from_df_to_nlist(df)
    real_temperatures = [float(row[0]) for row in last_5_days]

    today_temp = real_temperatures[0]
    t_recomendation = ""
    if today_temp < -15:
        t_recomendation = "На улице можно нос отморозить!"
    elif today_temp < -5 and today_temp >= -15:
        t_recomendation = "Самое время поиграть в снежки с друзьями!"
    elif today_temp < 10 and today_temp >= -5:
        t_recomendation = "Лучше заварить горячий чай и устроиться с книгой у окна"
    elif today_temp < 25 and today_temp >= 10:
        t_recomendation = "Идеальное время для прогулки на свежем воздухе"
    elif today_temp > 25:
        t_recomendation = "Самое время пойти искупаться!"

    today_wind_speed = int(sum(full_wind_speed) / len(full_wind_speed))
    w_recomendation = ""
    if today_wind_speed > 10:
        w_recomendation = "Ветер такой сильный, что даже голуби пешком ходят!"
    elif today_wind_speed > 3 and today_wind_speed <= 10:
        w_recomendation = "Не потеряй свою шляпу, на улице ветрено"
    elif today_wind_speed > 1 and today_wind_speed <= 3:
        w_recomendation = "Легкий ветерок создает прекрасную атмосферу для неспешной прогулки в парке"
    elif today_wind_speed >= 0 and today_wind_speed <= 1:
        w_recomendation = "Сегодня полный штиль, покататься на своей яхте не выйдет"

    if not df.empty:
        table = df24.to_html(classes='table table-striped', index=True, border=1)
        return render_template(
            'get-24hrs.html',
            city=city,
            full_dates=full_dates,
            full_temps=full_temps,
            today_temp=today_temp,
            t_recomendation=t_recomendation,
            w_recomendation=w_recomendation,
            table=table
        )

if __name__ == "__main__":
    model = load_trained_model('weather_model.pth')
    scaler_x = joblib.load('scaler_x.pkl')
    scaler_y = joblib.load('scaler_y.pkl')

    app.run(host='0.0.0.0', port=5000, debug=False)