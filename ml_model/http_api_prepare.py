import pandas as pd
import config
import pickle
import numpy as np
from config import model_folder


def get_history_ratings():
    ratings = pd.read_csv(config.data_folder + 'ratings.csv')
    movies_metadata = pd.read_csv(config.data_folder + 'movies_metadata.csv')
    movies_metadata = movies_metadata[['id', 'release_date', 'title']].copy()
    movies_metadata['id'] = movies_metadata['id'].apply(lambda x: -1 if str(x).__contains__('-') else int(x))
    movies_metadata['id'] = movies_metadata['id'].astype(np.int64)
    ratings = ratings.drop(columns='timestamp')
    ratings = ratings.merge(movies_metadata, how='left', left_on='movieId', right_on='id')
    history_ratings = ratings.drop(columns='id')
    pickle.dump(history_ratings, open(model_folder + 'history_ratings.pkl', 'wb'))


def load_movies_metadata():
    movies_metadata = pd.read_csv(config.data_folder + 'movies_metadata.csv')
    movies_metadata = movies_metadata[['id', 'release_date', 'title']].copy()
    movies_metadata['id'] = movies_metadata['id'].apply(lambda x: -1 if str(x).__contains__('-') else int(x))
    movies_metadata['id'] = movies_metadata['id'].astype(np.int64)
    pickle.dump(movies_metadata, open(model_folder + 'movies_metadata.pkl', 'wb'))


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
    pickle.dump(top_movie_ids, open(model_folder + 'top_movie_ids.pkl', 'wb'))


def prepare():
    get_history_ratings()
    load_movies_metadata()
    get_top_movies()
