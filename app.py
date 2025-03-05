from flask import Flask, request, jsonify
import joblib  # Load the trained model
import numpy as np
from some_text_processing_library import preprocess_text  # Replace with actual processing function

app = Flask(__name__)

# Load trained model
model = joblib.load("essay_rating_model.pkl")  # Ensure this file is in your deployment

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

        # Preprocess essay text (e.g., tokenize, clean)
        processed_essay = preprocess_text(essay_text)

        # Convert to model input format
        features = np.array([processed_essay])

        # Predict rating
        rating = model.predict(features)[0]

        return jsonify({"rating": rating})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
