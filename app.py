from flask import Flask, request, render_template
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.bot_interface import BotInterface

app = Flask(__name__)
bot = BotInterface()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form['query']
        response = bot.process_query(query)
        return render_template('index.html', query=query, response=response)
    return render_template('index.html', query='', response='')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
