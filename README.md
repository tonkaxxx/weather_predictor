# как запустить:

### через docker (рекомендуется)
1. скачать docker и docker compose
2. git clone git@github.com:tonkaxxx/weather_visualizer.git
3. сделать back/.env по образцу из back/example.env
4. docker compose up -d

### через исходный код
1. git clone git@github.com:tonkaxxx/weather_visualizer.git
2. сделать back/.env по образцу из back/example.env
3. pip install requests python-dotenv pandas scikit-learn joblib torch flask numpy --break-system-packages
4. python back/predictor.py
5. python main.py
6. зайти на http://127.0.0.1:5000