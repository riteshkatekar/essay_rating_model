from flask import Flask, request, jsonify
import joblib  # Load the trained model
import numpy as np
import os
import nltk
import textstat
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer



# Set NLTK data directory inside the project folder
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required NLTK resources
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('vader_lexicon', download_dir=nltk_data_path)

# nltk.download('punkt', download_dir='./nltk_data')
nltk.data.path.append("./nltk_data")


# Initialize NLP tools
sia = SentimentIntensityAnalyzer()

app = Flask(__name__)

# Load trained model
model_path = "essay_rating_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found!")

model = joblib.load(model_path)

# Feature extraction function
def extract_features(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    sentiment = sia.polarity_scores(text)
    word_count = len(words)
    sentence_count = len(sentences)
    lexical_diversity = len(set(words)) / word_count if word_count > 0 else 0
    flesch_reading_ease = textstat.flesch_reading_ease(text)

    return np.array([[sentiment['pos'], sentiment['neu'], sentiment['neg'], sentiment['compound'], 
                      word_count, sentence_count, lexical_diversity, flesch_reading_ease]])

@app.route("/")
def home():
    return "Essay Rating Model API is Running!"

@app.route("/rate_essay", methods=["POST"])
def rate_essay():
    try:
        data = request.json
        essay_text = data.get("essay", "").strip()

        if not essay_text:
            return jsonify({"error": "No essay provided"}), 400

        features = extract_features(essay_text)
        rating = model.predict(features)[0]

        return jsonify({"rating": int(rating)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Ensure Render detects the correct port
    app.run(host="0.0.0.0", port=port, debug=True)
