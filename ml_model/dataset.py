import numpy as np
import pandas as pd
from tqdm import tqdm


class DataSet:

    def __init__(self, ratings, batch_size=128):
        self.ratings = ratings
        self.batch_size = batch_size
        self.num_batch = self.ratings.shape[0] // self.batch_size

    def shuffle(self):
        np.random.shuffle(self.ratings)

    def get_batch(self, i):
        user_ids = self.ratings[i * self.batch_size: (i + 1) * self.batch_size, 0]
        item_ids = self.ratings[i * self.batch_size: (i + 1) * self.batch_size, 1]
        rates = self.ratings[i * self.batch_size: (i + 1) * self.batch_size, 2]
        return (np.array(user_ids, dtype=np.int32),
                np.array(item_ids, dtype=np.int32),
                np.array(rates, dtype=np.float32))


def test_dataset():
    base_folder = 'F:\\Projects\\train\\episerver\\data\\rs\\'
    ratings = pd.read_csv(base_folder + 'ratings.csv')
    ratings = ratings[['userId', 'movieId', 'rating']].values
    dataset = DataSet(ratings, batch_size=128)

    dataset.shuffle()
    for i in tqdm(range(dataset.num_batch)):
        user_ids, item_ids, ratings = dataset.get_batch(i)


if __name__ == '__main__':
    test_dataset()
