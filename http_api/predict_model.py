from threading import Lock
import pickle
import config
from config import model_folder
import numpy as np

lock = Lock()


def load_predict_user_dict():
    return pickle.load(open(config.model_folder + 'predict_user_dict.pkl', 'rb'))


def get_idx_movie_id_map():
    return pickle.load(open(config.model_folder + 'idx_movie_id_map.pkl', 'rb'))


class PredictModel:
    __instance = None

    def __new__(cls, *args, **kwargs):
        lock.acquire()
        if PredictModel.__instance is None:
            print('load ml_model first time')
            PredictModel.__instance = object.__new__(cls)
            PredictModel.__instance.history_ratings = pickle.load(open(model_folder + 'history_ratings.pkl', 'rb'))
            PredictModel.__instance.predict_user_dict = load_predict_user_dict()
            PredictModel.__instance.movies_metadata = pickle.load(open(model_folder + 'movies_metadata.pkl', 'rb'))
            PredictModel.__instance.top_movies = pickle.load(open(model_folder + 'top_movie_ids.pkl', 'rb'))
            PredictModel.__instance.idx_movie_id_map = get_idx_movie_id_map()
            print('done load ml_model first time')
        lock.release()
        return PredictModel.__instance

    def predict(self, user_id):
        rec_movie_ids = None
        is_new_user = False
        if self.predict_user_dict.__contains__(user_id):
            rec_item_idx = self.predict_user_dict[user_id]
            rec_movie_ids = []
            for item_idx in rec_item_idx:
                rec_movie_ids.append(self.idx_movie_id_map[item_idx])
        else:
            is_new_user = True
            rec_movie_ids = self.top_movies
        rec_movies = self.movies_metadata[self.movies_metadata['id'].isin(rec_movie_ids)][['title', 'release_date']].values
        np.random.shuffle(rec_movies)
        return is_new_user, rec_movies

    def get_history_movies(self, user_id):
        history_movies = self.history_ratings[self.history_ratings['userId'] == user_id][['title', 'release_date', 'rating']].values
        return history_movies


if __name__ == '__main__':
    predict_model = PredictModel()
