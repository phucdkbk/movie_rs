import ast
import pandas as pd
import numpy as np
from config import data_folder, model_folder, PreprocessConfig
from sklearn.model_selection import train_test_split
import pickle


def get_all_keywords(keywords_df):
    all_keywords = dict()
    for keyword in keywords_df['keywords']:
        list_keywords = ast.literal_eval(keyword)
        if list_keywords.__len__() > 0:
            for keyword_info in list_keywords:
                keyword_value = keyword_info['name']
                if not all_keywords.__contains__(keyword_value):
                    all_keywords[keyword_value] = 0
                all_keywords[keyword_value] += 1
    all_keywords = pd.DataFrame({
        'keyword': list(all_keywords.keys()),
        'num_count': list(all_keywords.values())
    })
    return all_keywords


def get_top_common_keywords(keyword_df):
    keyword_df = keyword_df.sort_values(by='num_count', ascending=False)
    common_keywords = keyword_df[keyword_df['num_count'] >= PreprocessConfig.MIN_OCCURRED_KEYWORD].keyword
    return common_keywords


def get_movie_keyword(common_keywords, keywords):
    map_keywords = {keyword: i + 1 for i, keyword in enumerate(common_keywords)}

    map_movie_keyword = dict()
    for movie_id, keyword in keywords[['id', 'keywords']].values:
        list_keywords = ast.literal_eval(keyword)
        movie_keywords = []
        if list_keywords.__len__() > 0:
            for keyword_info in list_keywords:
                keyword_value = keyword_info['name']
                if map_keywords.__contains__(keyword_value):
                    movie_keywords.append(map_keywords[keyword_value])
        map_movie_keyword[movie_id] = movie_keywords
    movie_keyword_df = pd.DataFrame({
        'movie_id': map_movie_keyword.keys(),
        'keywords': map_movie_keyword.values()
    })

    movie_keyword_df['num_keyword'] = movie_keyword_df['keywords'].apply(lambda x: len(x))
    return movie_keyword_df


def get_all_movie_keywords(all_movie_ids, movie_keyword_df, movie_id_idx_map):
    all_movie_keywords = pd.DataFrame({'movie_id': all_movie_ids})
    all_movie_keywords = all_movie_keywords.merge(movie_keyword_df, how='left', on='movie_id')
    all_movie_keywords['itemId'] = all_movie_keywords['movie_id'].map(movie_id_idx_map)
    all_movie_keywords = all_movie_keywords.sort_values(by='itemId', ascending=True)

    all_movie_keywords['num_keyword'] = all_movie_keywords['num_keyword'].fillna(0)
    all_movie_keywords['num_keyword'] = all_movie_keywords['num_keyword'].astype(np.int32)

    all_movie_keywords.loc[all_movie_keywords['padding_keywords'].isna(), 'padding_keywords'] = str(list(np.zeros(PreprocessConfig.MOVIE_MAX_KEYWORD, dtype=np.int32)))
    all_movie_keywords['padding_keywords'] = all_movie_keywords['padding_keywords'].apply(lambda x: np.array(ast.literal_eval(x)) if type(x) == str else x)

    item_keywords = np.concatenate(all_movie_keywords['padding_keywords'].values, axis=0).reshape(-1, 15)
    return item_keywords


def padding_keywords(movie_keyword_df, num_pad=PreprocessConfig.MOVIE_MAX_KEYWORD):
    map_padding_keywords = dict()
    for movie_id, keywords in movie_keyword_df[['movie_id', 'keywords']].values:
        if keywords.__len__() < num_pad:
            pad_keywords = np.concatenate([np.array(keywords), np.zeros(num_pad - keywords.__len__())], axis=0).astype(np.int32)
        else:
            pad_keywords = np.array(keywords[:15]).astype(np.int32)
        map_padding_keywords[movie_id] = pad_keywords
    movie_keyword_df['padding_keywords'] = movie_keyword_df['movie_id'].map(map_padding_keywords)
    movie_keyword_df = movie_keyword_df.drop(columns='keywords')
    return movie_keyword_df


def split_data(ratings):
    ratings = ratings.drop(columns=['timestamp', 'movieId'])
    train, val_test = train_test_split(ratings, test_size=0.1)
    val, test = train_test_split(val_test, test_size=0.5)
    return train, val, test


def get_common_keywords(keywords_df):
    all_keywords = get_all_keywords(keywords_df)
    common_keywords = get_top_common_keywords(all_keywords)
    return common_keywords


def preproces():
    print('----------  load data ---------')
    keywords = pd.read_csv(data_folder + 'keywords.csv')
    ratings = pd.read_csv(data_folder + 'ratings.csv')

    # map movie_ids to item_idx in range(0, NUM_ITEMS + 1)
    all_movie_ids = ratings['movieId'].unique()
    movie_id_idx_map = {movie_id: i for i, movie_id in enumerate(all_movie_ids)}
    idx_movie_id_map = {idx: movie_id for movie_id, idx in movie_id_idx_map.items()}

    # process keyword
    print('---------- process keyword ----------------------')
    common_keywords = get_common_keywords(keywords)
    movie_keyword_df = get_movie_keyword(common_keywords, keywords)
    movie_keyword_df = padding_keywords(movie_keyword_df)
    item_keywords = get_all_movie_keywords(all_movie_ids, movie_keyword_df, movie_id_idx_map)

    # train/val/test split
    print('---------- train/test/val split ----------------------')
    ratings['itemId'] = ratings['movieId'].map(movie_id_idx_map)
    train, val, test = split_data(ratings)

    # meta data
    meta_data = {
        'num_users': ratings['userId'].unique().__len__(),
        'num_items': ratings['itemId'].unique().__len__(),
        'num_keywords': common_keywords.__len__()
    }

    # save processed data
    print('---------- save processed data ----------------------')
    pickle.dump(train, open(model_folder + 'train.pkl', 'wb'))
    pickle.dump(val, open(model_folder + 'val.pkl', 'wb'))
    pickle.dump(test, open(model_folder + 'test.pkl', 'wb'))

    pickle.dump(movie_id_idx_map, open(model_folder + 'movie_id_idx_map.pkl', 'wb'))
    pickle.dump(idx_movie_id_map, open(model_folder + 'idx_movie_id_map.pkl', 'wb'))
    pickle.dump(common_keywords, open(model_folder + 'common_keywords.pkl', 'wb'))
    pickle.dump(item_keywords, open(model_folder + 'item_keywords.pkl', 'wb'))
    pickle.dump(meta_data, open(model_folder + 'meta_data.pkl', 'wb'))


if __name__ == '__main__':
    preproces()
