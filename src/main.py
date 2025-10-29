import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from azure.identity import DefaultAzureCredential
import argparse
import mlflow
import mlflow.sklearn
from azure.ai.ml import MLClient
from dotenv import load_dotenv
import os
import joblib, scipy.sparse as sp
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name for MLflow registration")
    parser.add_argument("--data", type=str, help="Data asset input")
    args = parser.parse_args()
    
    mlflow.start_run()
    mlflow.sklearn.autolog()
    
    data_asset = args.data

    mlflow.log_param("data_asset", data_asset)

    ratings = pd.read_csv(data_asset + "/ratings.csv")
    movies = pd.read_csv(data_asset + "/movies.csv")

    hot_encode_genres = pd.Series(movies['genres']).str.get_dummies(sep='|')
    encoded_movies = movies.merge(hot_encode_genres, left_index=True, right_index=True)
    encoded_movies.drop(columns=['genres'], inplace=True)

    ratings = ratings.drop(columns=['timestamp'])
    merged = ratings.merge(encoded_movies, on='movieId', how='left')

    df = merged.copy()

    df['liked'] = (df['rating'] >= 3).astype('int8')

    row_hash = pd.util.hash_pandas_object(df[['userId', 'movieId', 'liked']], index=False).astype('uint64')
    rand = (row_hash % (2**32)) / (2**32)

    df['split'] = np.where(rand < 0.5, 'train', 'test')
    
    train = df[df['split'] == 'train'].copy()
    test = df[df['split'] == 'test'].copy()

    user_encoder = OrdinalEncoder()
    movie_encoder = OrdinalEncoder()

    train['userId_enc'] = user_encoder.fit_transform(train[['userId']]).astype(np.int64)
    train['movieId_enc'] = movie_encoder.fit_transform(train[['movieId']]).astype(np.int64)

    known_users = train['userId'].unique()
    known_movies = train['movieId'].unique()

    test = test[test['userId'].isin(known_users) & test['movieId'].isin(known_movies)].copy()

    test['userId_enc'] = user_encoder.transform(test[['userId']]).astype(np.int64)
    test['movieId_enc'] = movie_encoder.transform(test[['movieId']]).astype(np.int64)

    train_liked = train[train['liked'] == 1]
    total_users = train['userId_enc'].max() + 1 
    total_movies = train['movieId_enc'].max() + 1

    X_train = csr_matrix(   
        (np.ones(train_liked.shape[0]), (train_liked['userId_enc'].to_numpy(), train_liked['movieId_enc'].to_numpy())),
        shape=(total_users, total_movies)
    )

    V_train = cosine_similarity(X_train.T, X_train.T, dense_output=False)
    
    seen_by_user = [set() for _ in range(total_users)]
    for u, m in zip(train['userId_enc'].to_numpy(), train['movieId_enc'].to_numpy()):
        seen_by_user[u].add(m)
            
    liked_by_user = {}

    for u, m, l in zip(test['userId_enc'].to_numpy(), test['movieId_enc'].to_numpy(), test['liked'].to_numpy()):
        if l == 1:
            liked_by_user.setdefault(u, set()).add(m)
            
    all_eval_users = np.array(list(liked_by_user.keys()), dtype=np.int64)
    h = pd.util.hash_array(all_eval_users)
    sample_mask = (h % 1000) == 0
    sample_users = set(all_eval_users[sample_mask])
    
    eval_results = evaluate(k=10, liked_by_user=liked_by_user, sample_users=sample_users)
    
    mlflow.log_metrics(eval_results)
    
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(exist_ok=True)
    
    joblib.dump(user_encoder, artifact_dir / "user_encoder.pkl")
    joblib.dump(movie_encoder, artifact_dir / "movie_encoder.pkl")
    sp.save_npz(artifact_dir / "similarities.npz", V_train)
    
    mlflow.log_artifacts(artifact_dir, artifact_path="artifacts")
    
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/artifacts"
    
    mlflow.register_model(model_uri, name=args.model_name)
    
    mlflow.end_run()
    
    
def get_recommendations_for_user(user_id_enc, top_k, seen_by_user, X_train, V_train):
    liked = [i for i in seen_by_user[user_id_enc] if X_train[user_id_enc, i] > 0]
    scores = V_train[:, liked].sum(axis=1).A.ravel()
    if seen_by_user[user_id_enc]:
        scores[list(seen_by_user[user_id_enc])] = -np.inf
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    return top_indices[np.argsort(-scores[top_indices])]

    
def evaluate(k, liked_by_user, sample_users):
    users_evaluated = 0
    users_with_hits = 0
    sum_precision = 0.0
    sum_recall = 0.0
    total_hits = 0
    total_liked = 0
    
    users = (u for u in liked_by_user.keys() if u in sample_users)
    
    for u in users:
        liked_movies = liked_by_user[u]
        print(f'Users evaluated: {users_evaluated}/{len(sample_users)}')
        recs = get_recommendations_for_user(u, top_k=k)
        if not recs.any():
            continue
        
        users_evaluated += 1
        num_hits = len(set(recs) & liked_movies)
        
        if num_hits > 0:
            users_with_hits += 1
            
        sum_precision += num_hits / k
        sum_recall += num_hits / len(liked_movies)
        
        total_hits += num_hits
        total_liked += len(liked_movies)
    
    precision = sum_precision / users_evaluated
    recall = sum_recall / users_evaluated
    hitrate = users_with_hits / users_evaluated
    
    
    return {
        'precision': precision,
        'recall': recall,
        'hitrate': hitrate,
        'total_hits': total_hits,
        'total_liked': total_liked
    }
    
if __name__ == "__main__":
    main()