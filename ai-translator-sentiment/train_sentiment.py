from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

texts = [
    "good","great","excellent","amazing","nice","happy","wonderful","love","like",
    "fantastic","pleasant","satisfied","best","brilliant",
    "I love this product","This is amazing","I am so happy today","What a wonderful experience",
    "I am satisfied with the service","This is really good","The result is excellent","Everything is great",
    "bad","terrible","awful","horrible","worst","sad","angry","disappointed",
    "This is the worst","I hate this thing","Very bad service","I am sad and disappointed",
    "I will never buy this again","This made me angry","This is a terrible experience",
    "The service was horrible"
]

labels = [
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression(max_iter=500)
model.fit(X, labels)

pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Sentiment model trained and saved.")
