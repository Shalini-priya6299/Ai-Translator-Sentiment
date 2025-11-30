from flask import Flask, render_template, request, jsonify
from langdetect import detect
from deep_translator import GoogleTranslator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle

app = Flask(__name__)

# Load ML model + vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

vader = SentimentIntensityAnalyzer()

language_names = {
    "es": "Spanish", "en": "English", "hi": "Hindi", "fr": "French",
    "de": "German", "bn": "Bengali", "ta": "Tamil", "pa": "Punjabi",
    "ur": "Urdu", "kn": "Kannada", "mr": "Marathi", "te": "Telugu"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()

    if text == "":
        return jsonify({"error": "Empty text"}), 400

    # Detect language
    try:
        lang_code = detect(text)
    except:
        lang_code = "unknown"

    language = language_names.get(lang_code, "Unknown")

    # Translate to English
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
    except:
        translated = text

    # Sentiment analysis
    score = vader.polarity_scores(translated)["compound"]
    if score > 0:
        sentiment = "Positive"
    elif score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return jsonify({
        "language": language,
        "translated": translated,
        "sentiment": sentiment
    })

if __name__ == "__main__":
    app.run(debug=True)
