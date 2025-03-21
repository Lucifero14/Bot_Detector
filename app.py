from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os
import pickle

app = Flask(__name__)

# Get the directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct paths for your model and scaler
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "chess_bot_model_1.pkl")

# Load the trained model and scaler
model_1 = joblib.load("models/chess_bot_model_1.pkl")
scaler = joblib.load("scaler.pkl")

# Threshold for direct bot detection when turns < 45
THRESHOLD_MILLISECONDS = 3000

# Serve the HTML file
@app.route("/")
def home():
    return render_template("index.html")  # Make sure index.html is inside a 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    turns = data["turns"]
    
    # White player data
    avg_move_time_white = data["avg_move_time_white"]
    rating_white = data["white_rating"]
    
    # Black player data
    avg_move_time_black = data["avg_move_time_black"]
    rating_black = data["black_rating"]
    
    # White player prediction
    if turns < 45:
        white_prediction = "Bot" if avg_move_time_white <= THRESHOLD_MILLISECONDS else "Human"
    else:
        normalized_move_time_white = avg_move_time_white / turns 
        white_prediction = "Bot" if model_1.predict([[normalized_move_time_white, rating_white]])[0] == 1 else "Human"
    
    # Black player prediction
    if turns < 45:
        black_prediction = "Bot" if avg_move_time_black <= THRESHOLD_MILLISECONDS else "Human"
    else:
        normalized_move_time_black = avg_move_time_black / turns 
        black_prediction = "Bot" if model_1.predict([[normalized_move_time_black, rating_black]])[0] == 1 else "Human"
    
    # Response
    response = {
        "White Player Prediction": white_prediction,
        "Black Player Prediction": black_prediction
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)