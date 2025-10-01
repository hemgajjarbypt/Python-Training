import requests
from dotenv import load_dotenv
import os
import json

def get_weather_data(city, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code}")

def save_weather_data(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def read_weather_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def main():
    load_dotenv()
    API_KEY = os.getenv('API_KEY')
    CITY = 'Ahmedabad'
    try:
        data = get_weather_data(CITY, API_KEY)
        file_path = os.path.join(os.getcwd(), "weather.json")
        save_weather_data(data, file_path)
        print("Weather data saved to weather.json")
        saved_data = read_weather_data(file_path)
        print("City:", saved_data["name"])
        print("Temperature:", saved_data["main"]["temp"])
        print("Weather:", saved_data["main"]["humidity"])
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()