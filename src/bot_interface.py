import sys
import os
import json
import re
import requests
from functools import lru_cache

# Adjust the system path to include the parent directory for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules (ensure these exist in your project structure)
from src.rag_system import RAGSystem
from src.model_inference import ModelInference
from src.crop_management import get_weather_data

# FAQ cache for quick greeting responses
faq_cache = {
    "hi": "Hi there! I'm Agribot. How can I assist you today?",
    "hello": "Hello!",
    "hey": "Hey there!",
    "hi there": "Hi there!",
    "good morning": "Good morning!",
    "good afternoon": "Good afternoon!",
    "good evening": "Good evening!"
}

class BotInterface:
    def __init__(self):
        """Initialize the BotInterface with necessary components."""
        self.rag = RAGSystem()
        self.model = ModelInference()
        self.indian_states = [
            "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
            "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
            "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
            "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
            "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi"
        ]
        self.weather_file = "data/weather_data.json"
        self.last_crop = None
        self.last_location = None
        self.last_response = None
        os.makedirs("data", exist_ok=True)
        self.check_ollama_server()

    def check_ollama_server(self):
        """Check if the Ollama server is running."""
        try:
            response = requests.get("http://localhost:11434", timeout=2)
            if response.status_code == 200:
                print("Ollama server is running.")
            else:
                print("Warning: Ollama server responded but may not be fully operational.")
        except requests.ConnectionError:
            print("Error: Ollama server is not running. Please start it with 'ollama serve'.")
        except requests.Timeout:
            print("Error: Ollama server check timed out. It might be slow or not running.")

    @lru_cache(maxsize=100)
    def cached_weather_data(self, location):
        """Fetch weather data with caching."""
        return get_weather_data(location)

    @lru_cache(maxsize=100)
    def cached_rag_retrieve(self, query, weather_str):
        """Retrieve RAG data with caching, converting weather_str back to dict."""
        weather = json.loads(weather_str)
        return self.rag.retrieve(query, weather)

    @lru_cache(maxsize=100)
    def cached_model_infer(self, prompt):
        """Perform model inference with caching."""
        return self.model.infer(prompt)

    def extract_location(self, query):
        """Extract an Indian state from the query."""
        query = query.lower()
        for state in self.indian_states:
            if state.lower() in query:
                return state
        return None

    def update_weather_data(self, location):
        """Update and cache weather data for a location."""
        if not location:
            return {"error": "No location specified"}
        weather = self.cached_weather_data(location)
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
        """Store the last response for context in subsequent queries."""
        self.last_response = {
            "query": query,
            "weather": weather,
            "rag": rag_response,
            "model": model_response
        }
        if weather and isinstance(weather, dict):
            self.last_location = self.extract_location(query) or self.last_location

    def process_query(self, query):
        """Process the user's query and return a response."""
        original_query = query
        query = query.lower()

        # Handle greetings (before main flow)
        contains_greeting = False
        for greeting in faq_cache.keys():
            if re.search(r'\b' + re.escape(greeting) + r'\b', query):
                contains_greeting = True
                query = re.sub(r'\b' + re.escape(greeting) + r'\b', '', query).strip()
                break
        if contains_greeting and not query:
            for greeting, response in faq_cache.items():
                if re.search(r'\b' + re.escape(greeting) + r'\b', original_query.lower()):
                    return response

        # Step 1: Check for location
        location = self.extract_location(query)
        if location and self.last_location and location != self.last_location:
            self.last_response = None
            self.last_location = location
        elif location:
            self.last_location = location

        # Crop suggestion patterns (location-based)
        crop_suggestion_patterns = [
            r"(?:what\s*(?:kind of|are the|is a good|is the best|can i use|suitable for|best|good)\s*crop[s]?|(?:best|suitable|good)\s*crop[s]?\s*for)\s*(?:in\s*)?(\w+)",
            r"what\s*(?:is|are)\s*the\s*best\s*crop[s]?\s*(?:to\s*use|to\s*grow|to\s*plant)?\s*(?:in|for)\s*(\w+)",
            r"what\s*crop[s]?\s*(?:should|can|must)\s*(?:i|we|one|you)?\s*(?:use|grow|plant)\s*(?:in|for)\s*(\w+)",
            r"what\s*(?:is|are)\s*the\s*crops?\s*that\s*(?:i|we|one|you)?\s*can\s*use\s*in\s*(\w+)"
        ]
        for pattern in crop_suggestion_patterns:
            crop_suggestion_match = re.search(pattern, query)
            if crop_suggestion_match:
                location_match = crop_suggestion_match.group(1)
                for state in self.indian_states:
                    if location_match.lower() in state.lower() or state.lower() in query:
                        location = state
                        return self.handle_crop_suggestion(location, original_query)
                break

        # Step 2: Check for specific crop if location exists
        if location:
            crop_match = re.search(r"(?:can |if |is )?(\w+)\s*(?:be used in|better for|good in|suitable for|for)\s*(\w+)", query)
            if not crop_match:
                for crop_name in [c["name"].lower() for c in self.rag.data.values() if c["category"] == "crops"]:
                    if crop_name in query:
                        crop = crop_name.capitalize()
                        break
                else:
                    crop = None
            else:
                crop = crop_match.group(1)
                location = location or crop_match.group(2)

            if crop:
                try:
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
                                response = (
                                    f"{crop} suits {location}'s weather well. "
                                    f"Its ideal temperature range is {temp_min}-{temp_max}°C. "
                                    f"Its humidity range is {humid_min}-{humid_max}%. "
                                    f"Current conditions ({temp}°C, {humidity}%) match perfectly. "
                                    f"It's a good choice for planting now."
                                )
                            elif temp_suitable:
                                response = (
                                    f"{crop} may work in {location}. "
                                    f"Its temperature range ({temp_min}-{temp_max}°C) fits {temp}°C. "
                                    f"However, humidity ({humidity}%) is below {humid_min}%. "
                                    f"Low humidity might stress the crop. "
                                    f"Consider irrigation to boost moisture."
                                )
                            elif humid_suitable:
                                response = (
                                    f"{crop} may not thrive in {location}. "
                                    f"Its humidity range ({humid_min}-{humid_max}%) fits {humidity}%. "
                                    f"But {temp}°C is outside {temp_min}-{temp_max}°C. "
                                    f"Temperature mismatch could affect growth. "
                                    f"Consider alternatives for better yield."
                                )
                            else:
                                response = (
                                    f"{crop} isn't ideal for {location}. "
                                    f"Its temperature range is {temp_min}-{temp_max}°C, not {temp}°C. "
                                    f"Its humidity range is {humid_min}-{humid_max}%, not {humidity}%. "
                                    f"These conditions don't match well. "
                                    f"Look for other crops instead."
                                )
                            full_response = f"Weather: {weather}\nRAG: {crop}\nModel: {response}"
                            self.set_last_response(query, weather, crop, response)
                            return full_response
                    response = f"Weather: {weather}\nRAG: No specific data found\nModel: {crop} is not in our database."
                    self.set_last_response(query, weather, "No specific data found", f"{crop} is not in our database.")
                    return response
                except Exception as e:
                    return f"Error fetching weather: {str(e)}"

        # Step 3: Check for "why" with previous context
        if re.search(r"\bwhy\b", query) and self.last_response:
            last_query = self.last_response["query"]
            weather = self.last_response["weather"]
            rag_response = self.last_response["rag"]
            last_model_response = self.last_response["model"]
            
            if self.last_crop and weather:  # Crop-specific "why"
                for crop_data in self.rag.data.values():
                    if crop_data["name"].lower() == self.last_crop.lower():
                        prefs = crop_data.get("weather_preferences", {})
                        temp_min = prefs.get("temperature", {}).get("min", -float('inf'))
                        temp_max = prefs.get("temperature", {}).get("max", float('inf'))
                        humid_min = prefs.get("humidity", {}).get("min", -float('inf'))
                        humid_max = prefs.get("humidity", {}).get("max", float('inf'))
                        temp = weather.get("temp", float('inf'))
                        humidity = weather.get("humidity", float('inf'))
                        prompt = (
                            f"Previous query: {last_query}\n"
                            f"Weather in {self.last_location}: {weather}\n"
                            f"RAG: {self.last_crop}\n"
                            f"Chatbot: Explain in exactly 5 short sentences why {self.last_crop} suits or doesn't suit "
                            f"{self.last_location}'s weather (temp: {temp}°C, humidity: {humidity}%), "
                            f"using its preferences (temp: {temp_min}-{temp_max}°C, humidity: {humid_min}-{humid_max}%)."
                        )
                        try:
                            model_response = self.cached_model_infer(prompt)
                            response = f"Weather: {weather}\nRAG: {self.last_crop}\nModel: {model_response}"
                            self.set_last_response(query, weather, self.last_crop, model_response)
                            return response
                        except ValueError as e:
                            return f"Error: {str(e)}"
                        except ConnectionError:
                            return "Error: Unable to connect to the model server. Please check if Ollama is running."
                return f"Model: I don't have enough data on {self.last_crop} to explain why."
            else:  # General "why" based on previous response
                prompt = (
                    f"Previous query: {last_query}\n"
                    f"Previous response: {last_model_response}\n"
                    f"Chatbot: Explain in exactly 5 short sentences why the previous response applies to '{last_query}'."
                )
                try:
                    model_response = self.cached_model_infer(prompt)
                    response = f"Model: {model_response}"
                    self.set_last_response(query, weather, rag_response, model_response)
                    return response
                except ValueError as e:
                    return f"Error: {str(e)}"
                except ConnectionError:
                    return "Error: Unable to connect to the model server. Please check if Ollama is running."

        # Step 4: General query if no location
        if not location:
            prompt = f"Chatbot: Respond to '{original_query}' as an agricultural bot in exactly 5 short sentences, no weather or RAG."
            try:
                model_response = self.cached_model_infer(prompt)
                response = f"Model: {model_response}"
                self.set_last_response(query, None, None, model_response)
                return response
            except ValueError as e:
                return f"Error: {str(e)}"  # E.g., "Model 'llama3' not found on Ollama server."
            except ConnectionError:
                return "Error: Unable to connect to the model server. Please check if Ollama is running."
            except Exception as e:
                return f"Error: Unexpected issue with model inference: {str(e)}"

        # Location-based catch-all (if location but no crop or specific pattern)
        try:
            weather = self.update_weather_data(location)
            if "error" in weather:
                return f"Error fetching weather for {location}: {weather['error']}"
            weather_str = json.dumps(weather, sort_keys=True)
            rag_response = self.cached_rag_retrieve(query, weather_str)
            if isinstance(rag_response, list) and rag_response:
                best_crop = rag_response[0]
                self.last_crop = best_crop
                prompt = (
                    f"Weather in {location}: {weather}\n"
                    f"RAG: {rag_response}\n"
                    f"Chatbot: Suggest {best_crop} as the best crop for {location}'s weather in exactly 5 short sentences "
                    f"based on weather comparison."
                )
            else:
                prompt = (
                    f"Weather in {location}: {weather}\n"
                    f"RAG: {rag_response}\n"
                    f"Chatbot: Suggest a suitable crop for {location}'s weather in exactly 5 short sentences based on weather data, "
                    f"noting no RAG match."
                )
            try:
                model_response = self.cached_model_infer(prompt)
                response = f"Weather: {weather}\nRAG: {rag_response}\nModel: {model_response}"
                self.set_last_response(query, weather, rag_response, model_response)
                return response
            except ValueError as e:
                return f"Error: {str(e)}"
            except ConnectionError:
                return "Error: Unable to connect to the model server. Please check if Ollama is running."
        except Exception as e:
            return f"Error fetching weather: {str(e)}"
    def handle_crop_suggestion(self, location, query):
        """Handle crop suggestion queries for a specific location."""
        if not location:
            return "Error: Unable to identify location in your query. Please specify a state in India."
        valid_location = False
        for state in self.indian_states:
            if location.lower() in state.lower() or state.lower() == location.lower():
                location = state
                valid_location = True
                break
        if not valid_location:
            return f"Error: '{location}' is not recognized as a valid location in India. Please specify a valid Indian state."
        
        try:
            weather = self.update_weather_data(location)
        except Exception as e:
            return f"Error fetching weather: {str(e)}"
        if "error" in weather:
            return f"Error: {weather['error']}"
        
        try:
            weather_str = json.dumps(weather, sort_keys=True)
            rag_response = self.cached_rag_retrieve(f"best crop for {location}", weather_str)
            if isinstance(rag_response, list) and rag_response:
                best_crop = rag_response[0]
                self.last_crop = best_crop
                prompt = (
                    f"Weather in {location}: {weather}\n"
                    f"RAG: {rag_response}\n"
                    f"Chatbot: Suggest {best_crop} as a suitable crop for {location}'s weather in exactly 5 short sentences "
                    f"based on weather comparison."
                )           
            else:
                prompt = (
                    f"Weather in {location}: {weather}\n"
                    f"RAG: {rag_response}\n"
                    f"Chatbot: Suggest a suitable crop for {location}'s weather in exactly 5 short sentences based on weather data, "
                    f"noting no RAG match."
                )
            try:
                model_response = self.cached_model_infer(prompt)
                response = f"Weather: {weather}\nRAG: {rag_response}\nModel: {model_response}"
                self.set_last_response(query, weather, rag_response, model_response)
                return response
            except ConnectionError:
                return "Error: Ollama server is not running. Please start it with 'ollama serve'."
        except Exception as e:
            return f"Error processing request: {str(e)}"

if __name__ == "__main__":
    bot = BotInterface()
    print(bot.process_query("can Crop1 be used in Punjab?"))
    print(bot.process_query("why"))
    print(bot.process_query("what is the best crop to use in Kerala?"))