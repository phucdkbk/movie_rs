import pandas as pd
from threading import Lock
import pickle
import config
import numpy as np

lock = Lock()


def get_ratings():
    ratings = pd.read_csv(config.data_folder + 'ratings.csv')
    movies_metadata = pd.read_csv(config.data_folder + 'movies_metadata.csv')
    movies_metadata = movies_metadata[['id', 'release_date', 'title']].copy()
    movies_metadata['id'] = movies_metadata['id'].apply(lambda x: -1 if str(x).__contains__('-') else int(x))
    movies_metadata['id'] = movies_metadata['id'].astype(np.int64)
    ratings = ratings.drop(columns='timestamp')
    ratings = ratings.merge(movies_metadata, how='left', left_on='movieId', right_on='id')
    ratings = ratings.drop(columns='id')
    return ratings


def load_predict_user_dict():
    return pickle.load(open(config.model_folder + 'predict_user_dict.pkl', 'rb'))


def load_movies_metadata():
    movies_metadata = pd.read_csv(config.data_folder + 'movies_metadata.csv')
    movies_metadata = movies_metadata[['id', 'release_date', 'title']].copy()
    movies_metadata['id'] = movies_metadata['id'].apply(lambda x: -1 if str(x).__contains__('-') else int(x))
    movies_metadata['id'] = movies_metadata['id'].astype(np.int64)
    return movies_metadata


def get_top_movies():
    ratings = pd.read_csv(config.data_folder + 'ratings.csv')
    movie_agg = ratings.groupby(by='movieId').agg({
        'userId': 'count',
        'rating': 'mean'
    })
    movie_agg.columns = ['num_ratings', 'mean_rating']
    movie_agg = movie_agg.reset_index()
    movie_agg = movie_agg[movie_agg['num_ratings'] > 1000]
    movie_agg = movie_agg.sort_values(by='mean_rating', ascending=False)
    top_movie_ids = list(movie_agg[:200]['movieId'])
    return top_movie_ids


def get_idx_movie_id_map():
    return pickle.load(open(config.model_folder + 'idx_movie_id_map.pkl', 'rb'))


class PredictModel:
    __instance = None

    def __new__(cls, *args, **kwargs):
        lock.acquire()
        if PredictModel.__instance is None:
            print('load ml_model first time')
            PredictModel.__instance = object.__new__(cls)
            PredictModel.__instance.ratings = get_ratings()
            PredictModel.__instance.predict_user_dict = load_predict_user_dict()
            PredictModel.__instance.movies_metadata = load_movies_metadata()
            PredictModel.__instance.top_movies = get_top_movies()
            PredictModel.__instance.idx_movie_id_map = get_idx_movie_id_map()
            print('done load ml_model first time')
        lock.release()
        return PredictModel.__instance

    def predict(self, user_id):
        rec_movie_ids = None
        if self.predict_user_dict.__contains__(user_id):
            rec_item_idx = self.predict_user_dict[user_id]
            rec_movie_ids = []
            for item_idx in rec_item_idx:
                rec_movie_ids.append(self.idx_movie_id_map[item_idx])
        else:
            rec_movie_ids = self.top_movies
        return self.movies_metadata[self.movies_metadata['id'].isin(rec_movie_ids)][['release_date', 'title']].values

    def get_history_movies(self, user_id):
        history_movie_ids = self.ratings[self.ratings['userId'] == user_id][['rating', 'release_date', 'title']].values
        return history_movie_ids


if __name__ == '__main__':
    predict_model = PredictModel()
