import requests
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Step 1: Call API
API_KEY = os.getenv('API_KEY')
CITY='Ahmedabad'
url=f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&units=metric&appid={API_KEY}"

response = requests.get(url)

if response.status_code == 200:
    data = response.json() 
    
    # Step 2: Save JSON to file
    file_path = os.path.join(os.getcwd(), "weather.json")
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
    
    print("Weather data saved to weather.json")
else:
    print("Error:", response.status_code)
    
# Step 3: Read data back from file
with open("weather.json", "r") as file:
    saved_data = json.load(file)
    
print("City:", saved_data["name"])
print("Temperature:", saved_data["main"]["temp"])
print("Weather:", saved_data["main"]["humidity"])