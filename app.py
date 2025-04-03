from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask("Application")

# Load dataset
df = pd.read_csv("dataset.csv")
df = df.drop_duplicates()
df['track_name'] = df['track_name'].fillna('Unknown').str.lower().str.strip()
df['artists'] = df['artists'].fillna('Unknown').str.lower().str.strip()
df['track_genre'] = df['track_genre'].fillna('Unknown').str.lower().str.strip()
df['combined_features'] = df['artists'] + ' ' + df['track_genre']

# TF-IDF and Nearest Neighbors
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'].fillna(''))
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(tfidf_matrix)

def recommend_songs(song_title, num_recommendations=5):
    song_title = song_title.lower().strip()
    print("Received song title:", song_title)
    print("Available songs:", df['track_name'].unique()[:10])  # Print some song names for debugging

    if song_title not in df['track_name'].str.lower().str.strip().tolist():
        
        return "Song not found in dataset"

    idx = df[df['track_name'].str.lower().str.strip() == song_title].index[0]

    distances, indices = model.kneighbors(tfidf_matrix[idx], n_neighbors=num_recommendations + 1)
    song_indices = indices.flatten()[1:]
    
    recommendations = df[['track_name', 'artists']].iloc[song_indices].values.tolist()
    
    print(recommendations)
    return recommendations


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    song = data.get('song', '').strip()
    
    recommendations = recommend_songs(song)
    return jsonify({'recommendations': [{'song': r[0], 'artist': r[1]} for r in recommendations]})

if __name__ == '__main__':
    app.run(debug=True)