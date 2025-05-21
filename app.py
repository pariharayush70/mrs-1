from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load all required files
with open("models/knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("models/title_to_id.pkl", "rb") as f:
    title_to_id = pickle.load(f)

with open("models/id_to_title.pkl", "rb") as f:
    id_to_title = pickle.load(f)

user_item_matrix = pd.read_pickle("models/user_item_matrix.pkl")

# Prepare sorted list of titles
movie_titles = sorted([
    title for title, mid in title_to_id.items()
    if mid in user_item_matrix.columns
])

# Recommendation function
def get_knn_recommendations(selected_title, top_n=5):
    if selected_title not in title_to_id:
        return []

    movie_id = title_to_id[selected_title]
    if movie_id not in user_item_matrix.columns:
        return []

    movie_vector = user_item_matrix[movie_id].values.reshape(1, -1)
    distances, indices = knn_model.kneighbors(movie_vector, n_neighbors=top_n + 1)
    recommended_ids = user_item_matrix.columns[indices.flatten()[1:]]
    recommended_titles = [id_to_title.get(mid, "Unknown") for mid in recommended_ids]
    return recommended_titles

# Route handler
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    selected_movie = None

    if request.method == "POST":
        selected_movie = request.form["movie"]
        recommendations = get_knn_recommendations(selected_movie)

    return render_template("index.html", movie_titles=movie_titles,
                           selected_movie=selected_movie,
                           recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
