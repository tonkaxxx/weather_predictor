import requests
import json
import os
from dotenv import load_dotenv


def get_data(city):
    load_dotenv()
    api_key = os.getenv("API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid="

    response = requests.get(f"{url}{api_key}")
    # проверка подключения
    if response.status_code == 200:
        json_data = json.dumps(response.json(), indent=4, ensure_ascii=False)
        print("conn est")
    else:
        print(f"bad conn - {response.status_code}")

    # переделываем данные в дикт
    data = json.loads(json_data)
    print(data)
    return data 

if __name__ == "__main__":
    get_data("london")
    
    

