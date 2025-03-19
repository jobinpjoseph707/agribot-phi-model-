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
        self.last_crop = None
        self.last_location = None
        self.last_response = None  # Store only the previous response
        os.makedirs("data", exist_ok=True)

    def extract_location(self, query):
        query_lower = query.lower()
        for state in self.indian_states:
            if state.lower() in query_lower:
                return state
        return None

    def update_weather_data(self, location):
        weather = get_weather_data(location)
        if "error" in weather:
            return weather
        if os.path.exists(self.weather_file):
            with open(self.weather_file, "r") as f:
                weather_data = json.load(f)
        else:
            weather_data = {}
        weather_data[location] = weather
        with open(self.weather_file, "w") as f:
            json.dump(weather_data, f)
        return weather

    def set_last_response(self, query, weather, rag_response, model_response):
        self.last_response = {
            "query": query,
            "weather": weather,
            "rag": rag_response,
            "model": model_response
        }
        if weather and isinstance(weather, dict):
            self.last_location = self.extract_location(query) or self.last_location

    def process_query(self, query):
        query_lower = query.lower()
        new_location = self.extract_location(query)

        # Reset last_response if location changes
        if new_location and self.last_location and new_location != self.last_location:
            self.last_response = None
            self.last_location = new_location
        elif new_location:
            self.last_location = new_location

        # Handle "why" based on last crop
        if re.search(r"\bwhy\b", query_lower) and self.last_crop and self.last_response:
            weather = self.last_response["weather"] or self.update_weather_data(self.last_location)
            if "error" in weather:
                return f"Error: {weather['error']}"
            for crop_data in self.rag.data.values():
                if crop_data["name"].lower() == self.last_crop.lower():
                    prefs = crop_data.get("weather_preferences", {})
                    temp_min = prefs.get("temperature", {}).get("min", -float('inf'))
                    temp_max = prefs.get("temperature", {}).get("max", float('inf'))
                    humid_min = prefs.get("humidity", {}).get("min", -float('inf'))
                    humid_max = prefs.get("humidity", {}).get("max", float('inf'))
                    temp = weather.get("temp", float('inf'))
                    humidity = weather.get("humidity", float('inf'))
                    prompt = f"Weather in {self.last_location}: {weather}\nRAG: {self.last_crop}\nChatbot: Explain in exactly 5 short sentences why {self.last_crop} suits or doesn’t suit {self.last_location}’s weather (temp: {temp}°C, humidity: {humidity}%), using its preferences (temp: {temp_min}-{temp_max}°C, humidity: {humid_min}-{humid_max}%)."
                    try:
                        model_response = self.model.infer(prompt)
                        response = f"Weather: {weather}\nRAG: {self.last_crop}\nModel: {model_response}"
                        self.set_last_response(query, weather, self.last_crop, model_response)
                        return response
                    except ConnectionError:
                        return "Error: Unable to connect to the model server. Please check if Ollama is running."
            return f"Model: I don’t have enough data on {self.last_crop} to explain why."

        # Handle planting instructions
        if re.search(r"how to\s*(use|plant|grow)\s*it", query_lower) and self.last_crop:
            location = self.last_location or new_location
            if location:
                weather = self.update_weather_data(location)
                if "error" in weather:
                    return f"Error: {weather['error']}"
                prompt = f"Weather in {location}: {weather}\nRAG: {self.last_crop}\nChatbot: Explain how to plant {self.last_crop} in {location}’s weather in exactly 5 short sentences."
                try:
                    model_response = self.model.infer(prompt)
                    response = f"Weather: {weather}\nRAG: {self.last_crop}\nModel: {model_response}"
                    self.set_last_response(query, weather, self.last_crop, model_response)
                    return response
                except ConnectionError:
                    return "Error: Unable to connect to the model server. Please check if Ollama is running."
            return f"Model: Please specify a location to explain how to plant {self.last_crop}."

        # Handle "what crop can be used" based on last location or new location
        if re.search(r"what\s*(else)?\s*crop(s)?\s*(can|to)?\s*(be)?\s*used", query_lower):
            location = new_location or self.last_location
            if location:
                weather = self.update_weather_data(location)
                if "error" in weather:
                    return f"Error: {weather['error']}"
                rag_response = self.rag.retrieve(f"best crop for {location}", weather)
                if isinstance(rag_response, list) and rag_response:
                    best_crop = rag_response[0]
                    self.last_crop = best_crop
                    prompt = f"Weather in {location}: {weather}\nRAG: {rag_response}\nChatbot: Suggest {best_crop} as a suitable crop for {location}’s weather in exactly 5 short sentences based on weather comparison."
                else:
                    prompt = f"Weather in {location}: {weather}\nRAG: {rag_response}\nChatbot: Suggest a suitable crop for {location}’s weather in exactly 5 short sentences based on weather data, noting no RAG match."
                try:
                    model_response = self.model.infer(prompt)
                    response = f"Weather: {weather}\nRAG: {rag_response}\nModel: {model_response}"
                    self.set_last_response(query, weather, rag_response, model_response)
                    return response
                except ConnectionError:
                    return "Error: Unable to connect to the model server. Please check if Ollama is running."
            return f"Model: Please specify a location to suggest a crop."

        # Detect crop and location in query
        crop_match = re.search(r"(?:can |if |is )?(\w+)\s*(?:be used in|better for|good in|suitable for|for)\s*(\w+)", query_lower)
        if not crop_match:
            for crop_name in [c["name"].lower() for c in self.rag.data.values() if c["category"] == "crops"]:
                if crop_name in query_lower:
                    crop = crop_name.capitalize()
                    location = new_location
                    break
            else:
                crop = None
                location = new_location
        else:
            crop = crop_match.group(1)
            location = new_location or crop_match.group(2)

        if crop and location:
            weather = self.update_weather_data(location)
            if "error" in weather:
                return f"Error: {weather['error']}"
            
            for crop_data in self.rag.data.values():
                if crop_data["category"] == "crops" and crop_data["name"].lower() == crop.lower():
                    self.last_crop = crop_data["name"]
                    self.last_location = location
                    prefs = crop_data.get("weather_preferences", {})
                    temp_min = prefs.get("temperature", {}).get("min", -float('inf'))
                    temp_max = prefs.get("temperature", {}).get("max", float('inf'))
                    humid_min = prefs.get("humidity", {}).get("min", -float('inf'))
                    humid_max = prefs.get("humidity", {}).get("max", float('inf'))
                    temp = weather.get("temp", float('inf'))
                    humidity = weather.get("humidity", float('inf'))
                    
                    temp_suitable = temp_min <= temp <= temp_max
                    humid_suitable = humid_min <= humidity <= humid_max
                    if temp_suitable and humid_suitable:
                        response = f"{crop} suits {location}’s weather well. Its ideal temperature range is {temp_min}-{temp_max}°C. Its humidity range is {humid_min}-{humid_max}%. Current conditions ({temp}°C, {humidity}%) match perfectly. It’s a good choice for planting now."
                    elif temp_suitable:
                        response = f"{crop} may work in {location}. Its temperature range ({temp_min}-{temp_max}°C) fits {temp}°C. However, humidity ({humidity}%) is below {humid_min}%. Low humidity might stress the crop. Consider irrigation to boost moisture."
                    elif humid_suitable:
                        response = f"{crop} may not thrive in {location}. Its humidity range ({humid_min}-{humid_max}%) fits {humidity}%. But {temp}°C is outside {temp_min}-{temp_max}°C. Temperature mismatch could affect growth. Consider alternatives for better yield."
                    else:
                        response = f"{crop} isn’t ideal for {location}. Its temperature range is {temp_min}-{temp_max}°C, not {temp}°C. Its humidity range is {humid_min}-{humid_max}%, not {humidity}%. These conditions don’t match well. Look for other crops instead."
                    
                    full_response = f"Weather: {weather}\nRAG: {crop}\nModel: {response}"
                    self.set_last_response(query, weather, crop, response)
                    return full_response

            response = f"Weather: {weather}\nRAG: No specific data found\nModel: {crop} is not in our database."
            self.set_last_response(query, weather, "No specific data found", f"{crop} is not in our database.")
            return response

        if not location:
            prompt = f"Chatbot: Respond to '{query}' as an agricultural bot in exactly 5 short sentences, no weather or RAG."
            try:
                model_response = self.model.infer(prompt)
                response = f"Model: {model_response}"
                self.set_last_response(query, None, None, model_response)
                return response
            except ConnectionError:
                return "Error: Unable to connect to the model server. Please check if Ollama is running."

        weather = self.update_weather_data(location)
        if "error" in weather:
            return f"Error fetching weather for {location}: {weather['error']}"

        rag_response = self.rag.retrieve(query, weather)
        if isinstance(rag_response, list) and rag_response:
            best_crop = rag_response[0]
            self.last_crop = best_crop
            self.last_location = location
            prompt = f"Weather in {location}: {weather}\nRAG: {rag_response}\nChatbot: Suggest {best_crop} as the best crop for {location}’s weather in exactly 5 short sentences based on weather comparison."
        else:
            prompt = f"Weather in {location}: {weather}\nRAG: {rag_response}\nChatbot: Suggest a suitable crop for {location}’s weather in exactly 5 short sentences based on weather data, noting no RAG match."
        try:
            model_response = self.model.infer(prompt)
            response = f"Weather: {weather}\nRAG: {rag_response}\nModel: {model_response}"
            self.set_last_response(query, weather, rag_response, model_response)
            return response
        except ConnectionError:
            return "Error: Unable to connect to the model server. Please check if Ollama is running."

if __name__ == "__main__":
    bot = BotInterface()
    print(bot.process_query("can Crop1 be used in Punjab?"))
    print(bot.process_query("why"))
    print(bot.process_query("so what crop can be used"))
    print(bot.process_query("can Crop1 be used in Kerala?"))