import ast
import pandas as pd
import numpy as np

MIN_OCCURRED_KEYWORD = 3
MOVIE_MAX_KEYWORD = 15


def get_all_keywords(keywords):
    all_keywords = dict()
    for keyword in keywords['keywords']:
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


def get_common_keywords(keyword_df):
    keyword_df = keyword_df.sort_values(by='num_count', ascending=False)
    common_keywords = keyword_df[keyword_df['num_count'] >= MIN_OCCURRED_KEYWORD].keyword
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


def get_all_movie_keywords(ratings, movie_keyword_df, movie_id_idx_map):
    all_movie_ids = ratings['movieId'].unique()
    all_movie_keywords = pd.DataFrame({'movie_id': all_movie_ids})
    all_movie_keywords = all_movie_keywords.merge(movie_keyword_df, how='left', on='movie_id')
    all_movie_keywords['itemId'] = all_movie_keywords['movie_id'].map(movie_id_idx_map)
    all_movie_keywords = all_movie_keywords.sort_values(by='itemId', ascending=True)

    all_movie_keywords['num_keyword'] = all_movie_keywords['num_keyword'].fillna(0)
    all_movie_keywords['num_keyword'] = all_movie_keywords['num_keyword'].astype(np.int32)

    all_movie_keywords.loc[all_movie_keywords['padding_keywords'].isna(), 'padding_keywords'] = str(list(np.zeros(MOVIE_MAX_KEYWORD, dtype=np.int32)))
    all_movie_keywords['padding_keywords'] = all_movie_keywords['padding_keywords'].apply(lambda x: np.array(ast.literal_eval(x)) if type(x) == str else x)

    item_keywords = np.concatenate(all_movie_keywords['padding_keywords'].values, axis=0).reshape(-1, 15)
    return item_keywords
