import toml
import ollama

def load_config():
    with open("config/config.toml", "r") as f:
        return toml.load(f)

class ModelInference:
    def __init__(self):
        config = load_config()
        self.model_path = config["settings"]["model_path"]
        self.client = ollama.Client(host="http://127.0.0.1:5051")  # Use port 5051

    def infer(self, input_text):
        response = self.client.generate(model=self.model_path, prompt=input_text)
        return response['response']
