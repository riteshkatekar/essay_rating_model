from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
with open("essay_rating_model.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route("/")
def home():
    return "Essay Rating Model API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]  # Expecting a list of features
    prediction = model.predict([np.array(data)])
    return jsonify({"rating": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
