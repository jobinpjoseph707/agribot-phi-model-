import requests
import toml
import os
import json

def load_config():
    with open("config/config.toml", "r") as f:
        return toml.load(f)

config = load_config()
WEATHER_API_KEY = config["api_keys"]["weather_api"]

def get_weather_data(location):
    cache_dir = "data/cache"
    cache_file = os.path.join(cache_dir, f"{location}_weather.json")
    os.makedirs(cache_dir, exist_ok=True)
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather = {
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "weather": data["weather"][0]["description"]
            }
            with open(cache_file, "w") as f:
                json.dump(weather, f)
            return weather
        else:
            return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}
