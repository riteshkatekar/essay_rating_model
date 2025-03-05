from flask import Flask, request, jsonify
import joblib  # Load the trained model
import numpy as np
import nltk
import textstat
import language_tool_python
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize Flask app
app = Flask(__name__)

# Download required NLTK resources
nltk.download('punkt')
nltk.download('vader_lexicon')

# Initialize NLP tools
sia = SentimentIntensityAnalyzer()
tool = language_tool_python.LanguageTool('en-US')

# Load trained model
model = joblib.load("essay_rating_model.pkl")  # Ensure this file is in your deployment

# Function to preprocess and extract features from text
def preprocess_text(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    sentiment = sia.polarity_scores(text)
    grammar_errors = len(tool.check(text))
    word_count = len(words)
    sentence_count = len(sentences)
    lexical_diversity = len(set(words)) / word_count if word_count > 0 else 0
    flesch_reading_ease = textstat.flesch_reading_ease(text)

    return [
        sentiment['pos'],
        sentiment['neu'],
        sentiment['neg'],
        sentiment['compound'],
        grammar_errors,
        word_count,
        sentence_count,
        lexical_diversity,
        flesch_reading_ease,
    ]

@app.route("/")
def home():
    return "Essay Rating Model API is Running!"

@app.route("/rate_essay", methods=["POST"])
def rate_essay():
    try:
        data = request.json
        essay_text = data.get("essay", "")

        if not essay_text:
            return jsonify({"error": "No essay provided"}), 400

        # Preprocess essay text
        processed_essay = preprocess_text(essay_text)
        features = np.array([processed_essay])

        # Predict rating
        rating = model.predict(features)[0]

        return jsonify({"rating": rating})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
