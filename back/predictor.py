import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

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

def generate_weather_data(days):
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

# разделяем данные на значение и лейбл
def split_data(data, past_size=5, forecast_horizon=3):
    x, y = [], []
    data_features = data[['temperature', 'humidity', 'pressure', 'wind_speed']].values
    
    # проходим по всему датасету
    for i in range(len(data) - past_size - forecast_horizon + 1):
        # значения: последние past_size дней
        past = data_features[i:(i + past_size)]
        # лейбл: следующие forecast_horizon дней
        future = data_features[i + past_size : i + past_size + forecast_horizon]
        
        x.append(past)
        y.append(future)
    
    return np.array(x), np.array(y)

class WeatherLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=12):
        super(WeatherLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        # инитим гейты лстм
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # прямой проход через лстм
        out, _ = self.lstm(x, (h0, c0))
        
        # берем только ласт выход
        out = out[:, -1, :]
        out = self.fc(out)
        
        # ретурним [batch_size, 3 дня, 4 параметра]
        return out.view(-1, 3, 4)

def predict_weather(model, last_days_data, scaler_x, scaler_y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    print(last_days_data)
    # нормализуем данные
    last_days_scaled = scaler_x.transform(last_days_data.reshape(-1, 4)).reshape(1, 5, 4)
    
    # юзаем torch.no_grad() чтобы не менялись веса
    with torch.no_grad():
        input_tensor = torch.FloatTensor(last_days_scaled).to(device)
        prediction = model(input_tensor)
    
    # обратно нормализуем
    prediction_np = prediction.cpu().numpy().reshape(-1, 4)
    prediction_original = scaler_y.inverse_transform(prediction_np).reshape(3, 4)
    
    return prediction_original

def load_trained_model(model_path='weather_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WeatherLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def save_model_and_scalers(model, scaler_x, scaler_y, model_path='weather_model.pth', 
                          scaler_x_path='scaler_x.pkl', scaler_y_path='scaler_y.pkl'):
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler_x, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=2) # это чтобы не было ешек в числах
    
    # генерируем данные
    weather_df = generate_weather_data(365*7)
    x, y = split_data(weather_df, past_size=5, forecast_horizon=3)

    # разделение на train/test
    train_size = int(0.8 * len(x))
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # скейлеры для нормализация данных
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    # решейпим массив из (365, 5, 4) -> (1825, 4)
    x_train_reshaped = x_train.reshape(-1, x_train.shape[-1])
    x_test_reshaped = x_test.reshape(-1, x_test.shape[-1])
    y_train_reshaped = y_train.reshape(-1, y_train.shape[-1])
    y_test_reshaped = y_test.reshape(-1, y_test.shape[-1])

    # нормализируем данные (от 0 до 1)
    x_train_scaled = scaler_x.fit_transform(x_train_reshaped).reshape(x_train.shape)
    x_test_scaled = scaler_x.transform(x_test_reshaped).reshape(x_test.shape)
    y_train_scaled = scaler_y.fit_transform(y_train_reshaped).reshape(y_train.shape)
    y_test_scaled = scaler_y.transform(y_test_reshaped).reshape(y_test.shape)

    # преобразовываем в тензоры 
    x_train_tensor = torch.FloatTensor(x_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    x_test_tensor = torch.FloatTensor(x_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)

    # создаем датасет и даталоадер (обязательно тест данные)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WeatherLSTM().to(device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # обучение модели
    num_epochs = 120
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # проходим вперед
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # проходим назад
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # валидациямодели
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_test_tensor.to(device))
            val_loss = criterion(val_outputs, y_test_tensor.to(device))
        
        train_losses.append(epoch_train_loss / len(train_loader))
        val_losses.append(val_loss.item())
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss/len(train_loader):.4f}, Val Loss: {val_loss.item():.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'early stopping at epoch {epoch}')
            # break

    model.load_state_dict(torch.load('best_model.pth'))

    # тестируем на реальных данных
    print("\n" + "="*50)
    print("ПРЕДСКАЗАНИЕ ПОГОДЫ НА 3 ДНЯ")
    print("="*50)

    test_sample = x_test[0]  # [5, 4] - 5 дней, 4 параметра
    true_future = y_test[0]  # [3, 4] - реальные значения на следующие 3 дня

    prediction = predict_weather(model, test_sample, scaler_x, scaler_y)

    parameters = ['температура', 'влажность', 'давление', 'скорость ветра']
    units = ['°C', '%', 'гПа', 'м/с']

    print("\nпервые 5 дней:")
    for i in range(5):
        print(f"день -{5-i}: " + ", ".join([f"{param}: {test_sample[i][j]:.1f}{unit}" 
        for j, (param, unit) in enumerate(zip(parameters, units))]))

    print("\nпредсказания на след 3 дня:")
    for i in range(3):
        print(f"день +{i+1}: " + ", ".join([f"{param}: {prediction[i][j]:.1f}{unit}" 
        for j, (param, unit) in enumerate(zip(parameters, units))]))

    print("\nреальные значения на след 3 дня:")
    for i in range(3):
        print(f"день +{i+1}: " + ", ".join([f"{param}: {true_future[i][j]:.1f}{unit}" 
        for j, (param, unit) in enumerate(zip(parameters, units))]))

    model.eval()
    with torch.no_grad():
        test_predictions = model(x_test_tensor.to(device))
        test_predictions = test_predictions.cpu().numpy()
        
        # обратно скейлим и решейпим
        test_predictions_original = scaler_y.inverse_transform(
            test_predictions.reshape(-1, 4)
        ).reshape(-1, 3, 4)
        
        y_test_original = scaler_y.inverse_transform(
            y_test_scaled.reshape(-1, 4)
        ).reshape(-1, 3, 4)

    # вычисляем мае 
    mae = np.mean(np.abs(test_predictions_original - y_test_original), axis=(0, 1))
    print("\nсредняя абсолютная ошибка по параметрам:")
    for param, error, unit in zip(parameters, mae, units):
        print(f"{param}: {error:.2f} {unit}")

    save_model_and_scalers(model, scaler_x, scaler_y)
