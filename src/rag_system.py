import os
import json
import re
from src.crop_management import get_weather_data

class RAGSystem:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.data = self.load_data()

    def load_data(self):
        data = {}
        # for category in ["crops", "diseases", "pests", "soil"]:
        for category in ["crops"]:
            dir_path = os.path.join(self.data_dir, category)
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    if file.endswith(".json"):
                        with open(os.path.join(dir_path, file), "r") as f:
                            item_data = json.load(f)
                            item_data["category"] = category
                            data[file[:-5]] = item_data
        return data

    def retrieve(self, query, weather):
        query_lower = query.lower()
        if re.search(r"best crop for|suitable crop for|recommended crop for|what is the best crop", query_lower):
            suitable_crops = []
            temp = weather.get("temp") if isinstance(weather, dict) else None
            humidity = weather.get("humidity") if isinstance(weather, dict) else None
            for key, value in self.data.items():
                if value.get("category") == "crops":
                    prefs = value.get("weather_preferences", {})
                    temp_min = prefs.get("temperature", {}).get("min", -float('inf'))
                    temp_max = prefs.get("temperature", {}).get("max", float('inf'))
                    humid_min = prefs.get("humidity", {}).get("min", -float('inf'))
                    humid_max = prefs.get("humidity", {}).get("max", float('inf'))
                    if (temp is not None and humidity is not None and 
                        temp_min <= temp <= temp_max and humid_min <= humidity <= humid_max):
                        suitable_crops.append(value["name"])
            return suitable_crops[:3] if suitable_crops else "No fully matching crops found in database."
        else:
            for key, value in self.data.items():
                if query_lower in key.lower() or any(query_lower in str(v).lower() for v in value.values()):
                    return value["name"] if value["category"] == "crops" else value
            return "No specific data found for " + query