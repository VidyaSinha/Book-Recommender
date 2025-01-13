from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load the trained model and data
with open('knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)

# Load the dataset
df = pd.read_csv('dataset.csv')
features = pickle.load(open('features.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return 'Song Recommendation System API is Running'

@app.route('/recommend', methods=['GET'])
def recommend():
    # Get the song name from the query parameters
    track_name = request.args.get('track_name')
    
    # Find the index of the track in the dataset
    idx = df[df['track_name'] == track_name].index
    
    if len(idx) == 0:
        return jsonify({'error': 'Track not found'}), 404

    idx = idx[0]

    # Get the nearest neighbors
    distances, indices = knn.kneighbors([features.iloc[idx]])

    # Get the top 'n' songs (excluding the input track itself)
    recommended_songs = [df.iloc[i]['track_name'] for i in indices[0][1:6]]
    
    # Return the recommended songs as JSON
    return jsonify(recommended_songs)

if __name__ == '__main__':
    app.run(debug=True)
