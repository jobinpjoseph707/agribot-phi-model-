from flask import Flask, request, render_template, jsonify
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.bot_interface import BotInterface

app = Flask(__name__)
bot = BotInterface()

# @app.after_request
# def add_csp_header(response):
#     response.headers['Content-Security-Policy'] = "script-src 'self' 'unsafe-eval';"
#     return response

@app.route('/', methods=['GET'])
def home():
    # Render the initial template with no chat history.
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def api_query():
    try:
        query = request.form['query']
        result = bot.process_query(query)

        # Check if the query contains any of the defined Indian states.
        if any(state.lower() in query.lower() for state in bot.indian_states):
            # For location-based queries, assume the response starts with "Weather:".
            lines = result.split('\n')
            weather = ""
            if lines and lines[0].startswith("Weather:"):
                weather = lines[0].replace("Weather:", "").strip()
            return jsonify(response=result, weather=weather)
        else:
            # For general queries, return only the bot's response.
            return jsonify(response=result)
    except KeyError as e:
        return jsonify({"error": f"Missing form data: {e}"}), 400
    except Exception as e:
        print(f"Error processing query: {e}")  # Log the error for debugging
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
