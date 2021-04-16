from http_api import http_demo
import argparse
from ml_model import data_preprocessing, rec_model
import tensorflow as tf
import logging
from logging.handlers import RotatingFileHandler


def init_logger():
    logger = logging.getLogger('movie_rs')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    rotate_handler = RotatingFileHandler('./logs/movie_rs.log', maxBytes=200000000, backupCount=10)
    rotate_handler.setLevel(logging.INFO)
    rotate_handler.setFormatter(formatter)

    logger.addHandler(rotate_handler)
    logger.addHandler(ch)


def parse_args():
    parser = argparse.ArgumentParser(description="Run movie recommendation.")
    parser.add_argument('--action', nargs='?', default='http_demo',
                        help='Action to do.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size for users and items.')
    parser.add_argument('--keyword_embed_size', type=int, default=64,
                        help='Embedding size for keywords.')
    parser.add_argument('--alpha', type=float, default=0.005,
                        help='Regularization for keyword embeddings.')
    parser.add_argument('--beta', type=float, default=0.005,
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    init_logger()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus.__len__() > 0:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    if args.action == 'http_demo':
        http_demo.start_api()
    elif args.action == 'preprocess':
        data_preprocessing.preproces()
    elif args.action == 'train':
        rec_model.train_and_predict(args)
