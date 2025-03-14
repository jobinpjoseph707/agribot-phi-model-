import sys
import os
import json
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag_system import RAGSystem
from src.model_inference import ModelInference
from src.crop_management import get_weather_data

class BotInterface:
    def __init__(self):
        self.rag = RAGSystem()
        self.model = ModelInference()
        self.indian_states = ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi"]
        self.weather_file = "data/weather_data.json"
        os.makedirs("data", exist_ok=True)

    def extract_location(self, query):
        query_lower = query.lower()
        for state in self.indian_states:
            if state.lower() in query_lower:
                return state
        return None  # No location detected

    def update_weather_data(self, location):
        weather = get_weather_data(location)
        if "error" in weather:
            return weather
        # Load existing weather data or create new
        if os.path.exists(self.weather_file):
            with open(self.weather_file, "r") as f:
                weather_data = json.load(f)
        else:
            weather_data = {}
        weather_data[location] = weather
        with open(self.weather_file, "w") as f:
            json.dump(weather_data, f)
        return weather

    def process_query(self, query):
        location = self.extract_location(query)
        if location:  # Location detected, use RAG and weather
            weather = self.update_weather_data(location)
            if "error" in weather:
                return f"Error fetching weather for {location}: {weather['error']}"
            rag_response = self.rag.retrieve(query, weather)
            prompt = f"Weather in {location}: {weather}\nRAG: {rag_response}\nChatbot: Pick one crop from RAG and suggest it for {location}s weather in exactly 5 short sentences."
            model_response = self.model.infer(prompt)
            return f"Weather: {weather}\nRAG: {rag_response}\nModel: {model_response}"
        else:  # No location, plain chat
            prompt = f"Chatbot: Respond to '{query}' in a friendly way, no weather or RAG, max 5 short sentences."
            model_response = self.model.infer(prompt)
            return f"Model: {model_response}"

if __name__ == "__main__":
    bot = BotInterface()
    print(bot.process_query("What is best crop for Punjab weather?"))
