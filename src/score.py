import joblib
import numpy as np
import pandas as pd
import json
import os

def init():
    global user_enc, movie_enc, movie_indices, movie_similarities, movies, seen_by_user, liked_by_users
    
    model_dir = os.getenv('AZUREML_MODEL_DIR')

    user_enc = joblib.load(f'{model_dir}/artifacts/user_encoder.pkl')
    movie_enc = joblib.load(f'{model_dir}/artifacts/movie_encoder.pkl')
    
    movie_indices = np.load(f'{model_dir}/artifacts/movie_knn_indices.npy')
    movie_similarities = np.load(f'{model_dir}/artifacts/movie_knn_similarities.npy')
    movies = pd.read_parquet(f'{model_dir}/artifacts/movies.parquet')
    seen_by_user = joblib.load(f'{model_dir}/artifacts/seen_by_user.pkl')
    liked_by_users = joblib.load(f'{model_dir}/artifacts/liked_by_users.pkl')    
    

def recommend_movie_to_user(user_id, top_k=10):
    user_id_enc = user_enc.transform([[user_id]]).astype(np.int64)[0, 0]
    
    liked = [i for i in liked_by_users[user_id_enc]]
    
    scores = np.zeros(len(movie_indices), dtype=np.float32)
    neighbors = movie_indices[liked].ravel()
    weights = movie_similarities[liked].ravel()
    np.add.at(scores, neighbors, weights)
    
    scores[list(seen_by_user[user_id_enc])] = -np.inf
    
    top = np.argpartition(scores, -top_k)[-top_k:]
    rec_idxs_enc = top[np.argsort(-scores[top])]
    
    rec_idxs_raw = movie_enc.inverse_transform(rec_idxs_enc.reshape(-1, 1)).ravel()
    
    recs = movies[movies['movieId'].isin(rec_idxs_raw)][['movieId', 'title', 'genres']]
    return recs.to_dict(orient='records')


def run(raw_data):
    try:
        data = json.loads(raw_data)
        user_id = data['userId']
        top_k = data.get('top_k', 10)
        recommendations = recommend_movie_to_user(user_id, top_k=top_k)
        return json.dumps({'recommendations': recommendations})
    except Exception as e:
        return json.dumps({"error": str(e)})