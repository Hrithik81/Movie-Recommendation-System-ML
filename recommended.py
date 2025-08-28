from flask import Flask, render_template, request
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# -------------------------
# CONFIG
# -------------------------
API_KEY = "642332f3"   # Your OMDb API key
CSV_FILE = "movies.csv"

app = Flask(__name__)

# -------------------------
# NLTK setup
# -------------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# -------------------------
# Load and preprocess data
# -------------------------
df = pd.read_csv(CSV_FILE).dropna().reset_index(drop=True)
df["combined"] = df["genres"] + " " + df["keywords"] + " " + df["overview"]
data = df[["title", "combined"]].copy()

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

data["cleaned_text"] = data["combined"].apply(preprocess_text)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(data["cleaned_text"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# -------------------------
# Recommendation function
# -------------------------
def recommended_movies(movie_name, cosine_sim=cosine_sim, df=data, top_n=5):
    matches = df[df["title"].str.lower().str.contains(movie_name.lower())]
    if matches.empty:
        return []
    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1 : top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    return df["title"].iloc[movie_indices]

# -------------------------
# OMDb API
# -------------------------
def fetch_movie_details(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get("Response") == "True":
            return {
                "Title": data.get("Title"),
                "Year": data.get("Year"),
                "Genre": data.get("Genre"),
                "Plot": data.get("Plot"),
                "Poster": data.get("Poster"),
                "IMDB Rating": data.get("imdbRating"),
            }
    return None

# -------------------------
# Flask Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    movie_name = ""
    if request.method == "POST":
        movie_name = request.form["movie"]
        recs = recommended_movies(movie_name)
        for rec in recs:
            details = fetch_movie_details(rec)
            if details:
                recommendations.append(details)
    return render_template("index.html", movie=movie_name, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
