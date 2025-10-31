import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OrdinalEncoder, normalize
from sklearn.neighbors import NearestNeighbors
import numpy as np
import argparse
import mlflow
import joblib
from pathlib import Path


def main():
    def get_recommendations_for_user(user_id_enc, top_k=10):
        liked = [i for i in liked_by_users[user_id_enc]]
    
        scores = np.zeros(len(indices), dtype=np.float32)
        neighbors = indices[liked].ravel()
        weights = sims[liked].ravel()
        np.add.at(scores, neighbors, weights)
        
        scores[list(seen_by_user[user_id_enc])] = -np.inf
        
        top = np.argpartition(scores, -top_k)[-top_k:]
        recommended_indices = top[np.argsort(-scores[top])]
        return recommended_indices
    
    def evaluate(k):
        users_evaluated = 0
        users_with_hits = 0
        sum_precision = 0.0
        sum_recall = 0.0
        total_hits = 0
        total_liked = 0
        
        users = (u for u in liked_by_users.keys() if u in sample_users)
        
        for u in users:
            liked_movies = liked_by_users[u]
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
        
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name for MLflow registration")
    parser.add_argument("--data", type=str, help="Data asset input")
    args = parser.parse_args()
    
    mlflow.start_run()
    
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

    max_users = 100_000
    sampled_users = np.random.choice(df['userId'].unique(), size=max_users, replace=False)
    df = df[df['userId'].isin(sampled_users)].copy()
        
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
        (np.ones(train_liked.shape[0], dtype=np.float32), (train_liked['userId_enc'].to_numpy(), train_liked['movieId_enc'].to_numpy())),
        shape=(total_users, total_movies),
        dtype=np.float32
    )

    K = 100
    X_movies = X_train.T.tocsr().astype(np.float32)
    X_movies = normalize(X_movies, axis=1, copy=False)

    nn = NearestNeighbors(n_neighbors=K+1, metric='cosine', algorithm='brute', n_jobs=-1)
    nn.fit(X_movies)
    
    dists, indices = nn.kneighbors(X_movies, return_distance=True, n_neighbors=K+1)
    sims = 1.0 - dists
    
    indices = indices[:, 1:]
    sims = sims[:, 1:].astype(np.float32)
    
    seen_by_user = [set() for _ in range(total_users)]
    for u, m in zip(train['userId_enc'].to_numpy(), train['movieId_enc'].to_numpy()):
        seen_by_user[u].add(m)
            
    liked_by_users = {}

    for u, m, l in zip(test['userId_enc'].to_numpy(), test['movieId_enc'].to_numpy(), test['liked'].to_numpy()):
        if l == 1:
            liked_by_users.setdefault(u, set()).add(m)
            
    all_eval_users = np.array(list(liked_by_users.keys()), dtype=np.int64)
    h = pd.util.hash_array(all_eval_users)
    sample_mask = (h % 1000) == 0
    sample_users = set(all_eval_users[sample_mask])
    
    eval_results = evaluate(k=10)
    
    mlflow.log_metrics(eval_results)
    
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(exist_ok=True)
    
    joblib.dump(user_encoder, artifact_dir / "user_encoder.pkl")
    joblib.dump(movie_encoder, artifact_dir / "movie_encoder.pkl")
    joblib.dump(seen_by_user, artifact_dir / "seen_by_user.pkl")
    joblib.dump(liked_by_users, artifact_dir / "liked_by_users.pkl")
    np.save(artifact_dir / "movie_knn_indices.npy", indices.astype(np.int32))
    np.save(artifact_dir / "movie_knn_similarities.npy", sims.astype(np.float32))
    movies.to_parquet(artifact_dir / "movies.parquet", index=False)
    
    mlflow.log_artifacts(str(artifact_dir), artifact_path="artifacts")
    
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/artifacts"
    
    mlflow.register_model(model_uri, name=args.model_name)
    
    mlflow.end_run()
    
    
    
if __name__ == "__main__":
    main()