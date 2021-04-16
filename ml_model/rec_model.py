import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
from tensorflow.keras.initializers import TruncatedNormal
from tqdm import tqdm
from time import time
import pickle
from config import data_folder, model_folder
from ml_model.dataset import DataSet


class RSModel(Model):

    def __init__(self, args):
        super(RSModel, self).__init__()
        self.embedding_size = args['embedding_size']
        self.keyword_embedding_size = args['keyword_embedding_size']
        self.alpha = args['alpha']
        self.beta = args['beta']
        self.gamma = args['gamma']
        self.num_items = args['num_items']
        self.num_users = args['num_users']
        self.num_keywords = args['num_keywords']
        self.item_keywords = tf.constant(args['item_keywords'], dtype=tf.int32)
        self.keyword_embedding = tf.keras.layers.Embedding(input_dim=self.num_keywords + 1, output_dim=self.keyword_embedding_size,
                                                           embeddings_initializer=TruncatedNormal(mean=0., stddev=0.1),
                                                           mask_zero=True,
                                                           embeddings_regularizer=tf.keras.regularizers.L2(self.alpha)
                                                           )
        self.user_embedding = tf.keras.layers.Embedding(input_dim=self.num_users + 1, output_dim=self.embedding_size,
                                                        embeddings_initializer=TruncatedNormal(mean=0., stddev=0.1),
                                                        embeddings_regularizer=tf.keras.regularizers.L2(self.beta))
        self.item_embedding = tf.keras.layers.Embedding(input_dim=self.num_items, output_dim=self.embedding_size,
                                                        embeddings_initializer=TruncatedNormal(mean=0., stddev=0.1),
                                                        embeddings_regularizer=tf.keras.regularizers.L2(self.beta))
        self.bias_u = tf.keras.layers.Embedding(input_dim=self.num_users + 1, output_dim=1,
                                                embeddings_initializer=TruncatedNormal(mean=0., stddev=0.1),
                                                embeddings_regularizer=tf.keras.regularizers.L2(self.gamma))
        self.bias_i = tf.keras.layers.Embedding(input_dim=self.num_items, output_dim=1,
                                                embeddings_initializer=TruncatedNormal(mean=0., stddev=0.1),
                                                embeddings_regularizer=tf.keras.regularizers.L2(self.gamma))
        self.mlp_dense = tf.keras.layers.Dense(units=1)

    def call(self, user_ids, item_ids):
        user_bias = self.bias_u(user_ids)
        item_bias = self.bias_i(item_ids)
        # matrix factorization
        users_embedding = self.user_embedding(user_ids)
        items_embedding = self.item_embedding(item_ids)
        mf = tf.math.multiply(users_embedding, items_embedding)
        # mlp
        item_keyword = tf.nn.embedding_lookup(self.item_keywords, item_ids)
        item_keyword_embedding = self.keyword_embedding(item_keyword)
        item_encode = tf.reduce_sum(item_keyword_embedding, axis=1)
        item_encode = self.mlp_dense(item_encode)
        # rating score
        r = tf.squeeze(user_bias) + tf.squeeze(item_bias) + tf.reduce_sum(mf, axis=1) + tf.reduce_sum(item_encode, axis=1)

        #         r = tf.squeeze(user_bias) + tf.squeeze(item_bias) + tf.reduce_sum(mf, axis=1)
        return r

    def loss_fn_rmse(self, predictions, labels):
        loss = tf.reduce_sum(tf.math.square(predictions - labels))
        loss += tf.reduce_sum(self.keyword_embedding.losses)
        loss += tf.reduce_sum(self.user_embedding.losses) + tf.reduce_sum(self.item_embedding.losses)
        #         loss += tf.reduce_sum(self.bias_u.losses) + tf.reduce_sum(self.bias_i.losses)
        return loss


@tf.function
def train_step(rs_model, optimizer, user_ids, item_ids, ratings):
    with tf.GradientTape() as tape:
        predictions = rs_model(user_ids, item_ids)
        loss = rs_model.loss_fn_rmse(predictions, ratings)
    gradients = tape.gradient(target=loss, sources=rs_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, rs_model.trainable_variables))
    return loss


def get_val_rmse(rs_model, val_dataset):
    all_ratings = []
    all_predictions = []
    for i in tqdm(range(val_dataset.num_batch)):
        user_ids, item_ids, ratings = val_dataset.get_batch(i)
        predictions = rs_model(user_ids, item_ids)
        all_predictions.append(predictions.numpy())
        all_ratings.append(ratings)
    val_predictions = np.concatenate(all_predictions, axis=0)
    val_ratings = np.concatenate(all_ratings, axis=0)
    return np.sqrt(np.mean((val_predictions - val_ratings) ** 2))


def training(rs_model, optimizer, train_dataset, val_dataset, num_epochs, pretrained=False):
    epoch_step = tf.Variable(0, dtype=tf.int32)
    ckpt = tf.train.Checkpoint(rec_model=rs_model, epoch_step=epoch_step)
    manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=model_folder + 'rsmodel_ckpt', max_to_keep=3)
    if pretrained:
        ckpt.restore(manager.latest_checkpoint)
    for epoch in range(num_epochs):
        train_loss = tf.constant(0, tf.float32)
        start_load_data = time()
        train_dataset.shuffle()
        load_data_time = time() - start_load_data
        start_train_time = time()
        for i in tqdm(range(train_dataset.num_batch)):
            user_ids, item_ids, ratings = train_dataset.get_batch(i)
            loss_step = train_step(rs_model, optimizer, user_ids, item_ids, ratings)
            train_loss += loss_step
            # because training on full epoch takes time, break and monitor score
            if i > 2000:
                break
        train_time = time() - start_train_time
        print('epoch: ', epoch, '. load data time: ', load_data_time, '. train time: ', train_time, '. train loss: ', train_loss.numpy())
        if epoch % 2 == 0:
            val_rmse = get_val_rmse(rs_model, val_dataset)
            score = {'val_rmse': val_rmse}

            print('epoch: {}, score: {}'.format(epoch, score))
            ckpt.epoch_step.assign_add(epoch + 1)
            manager.save()
            print('done save at epoch: ', ckpt.epoch_step.numpy())


def train_model(input_args):
    train = pickle.load(open(model_folder + 'train.pkl', 'rb'))
    val = pickle.load(open(model_folder + 'val.pkl', 'rb'))

    meta_data = pickle.load(open(model_folder + 'meta_data.pkl', 'rb'))

    item_keywords = pickle.load(open(model_folder + 'item_keywords.pkl', 'rb'))

    train_dataset = DataSet(train[['userId', 'itemId', 'rating']].values, batch_size=1024)
    val_dataset = DataSet(val[['userId', 'itemId', 'rating']].values, batch_size=1024)

    args = dict()
    args['embedding_size'] = input_args.embed_size
    args['keyword_embedding_size'] = input_args.keyword_embed_size
    args['alpha'] = 0.005
    args['beta'] = 0.005
    args['gamma'] = 0.000
    args['num_items'] = meta_data['num_items']
    args['num_users'] = meta_data['num_users']
    args['num_keywords'] = meta_data['num_keywords']
    args['item_keywords'] = item_keywords

    rsmodel = RSModel(args)
    opt = tf.keras.optimizers.Adam(learning_rate=input_args.lr)
    training(rsmodel, opt, train_dataset, val_dataset, num_epochs=input_args.epochs)


def predict():
    # load pretrain model
    test = pickle.load(open(model_folder + 'test.pkl', 'rb'))

    meta_data = pickle.load(open(model_folder + 'meta_data.pkl', 'rb'))
    item_keywords = pickle.load(open(model_folder + 'item_keywords.pkl', 'rb'))

    args = dict()
    args['embedding_size'] = 64
    args['keyword_embedding_size'] = 64
    args['alpha'] = 0.005
    args['beta'] = 0.005
    args['gamma'] = 0.000
    args['num_items'] = meta_data['num_items']
    args['num_users'] = meta_data['num_users']
    args['num_keywords'] = meta_data['num_keywords']
    args['item_keywords'] = item_keywords

    rec_model = RSModel(args)

    epoch_step = tf.Variable(0, dtype=tf.int32)
    ckpt = tf.train.Checkpoint(rec_model=rec_model, epoch_step=epoch_step)
    manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=model_folder + 'rsmodel_ckpt', max_to_keep=3)
    print('load pretrained model at: ' + manager.latest_checkpoint)
    ckpt.restore(manager.latest_checkpoint)

    test_dataset = DataSet(test[['userId', 'itemId', 'rating']].values, batch_size=1024)
    test_score = get_val_rmse(rec_model, test_dataset)
    print("test score: ", test_score)

    # prepare data for prediction
    top_n = 100
    item_ids = np.array(range(meta_data['num_items']))
    users_embedding = rec_model.user_embedding.weights[0].numpy()
    items_encode = rec_model.item_embedding.weights[0].numpy()
    items_bias = np.squeeze(rec_model.bias_i.weights[0].numpy())
    users_bias = rec_model.bias_u.weights[0].numpy()

    item_keyword = tf.nn.embedding_lookup(rec_model.item_keywords, item_ids)
    item_keyword_embedding = rec_model.keyword_embedding(item_keyword)
    item_keyword_encode = tf.reduce_sum(item_keyword_embedding, axis=1)
    item_keyword_encode = rec_model.mlp_dense(item_keyword_encode)
    item_keyword_encode = np.squeeze(item_keyword_encode.numpy())

    # predict top_n for user
    predict_user_dict = dict()
    for user_id in tqdm(range(1, meta_data['num_users'])):
        user_embedded = users_embedding[user_id]
        user_bias = users_bias[user_id]
        predicts = np.squeeze(np.matmul(user_embedded.reshape(1, -1), items_encode.T)) + items_bias + user_bias + item_keyword_encode
        best = np.argpartition(predicts, -top_n)[-top_n:]
        predict_user_dict[user_id] = best.astype(np.int32)

    # save top_n user
    pickle.dump(predict_user_dict, open(model_folder + 'predict_user_dict.pkl', 'wb'))


def train_and_predict(input_args):
    train_model(input_args)
    predict()


if __name__ == '__main__':
    train_and_predict()
