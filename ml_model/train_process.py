import tensorflow as tf
import numpy as np
from tqdm import tqdm
from time import time
import pickle
from config import model_folder
from ml_model.dataset import DataSet
from ml_model import http_api_prepare
from ml_model.rec_model import RSModel_7
import config


@tf.function
def train_step(rs_model, optimizer, user_ids, item_ids, ratings):
    with tf.GradientTape() as tape:
        predictions = rs_model(user_ids, item_ids)
        loss, rmse = rs_model.loss_fn_rmse(predictions, ratings)
    gradients = tape.gradient(target=loss, sources=rs_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, rs_model.trainable_variables))
    return loss, rmse


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
        train_rmse = tf.constant(0, tf.float32)
        start_load_data = time()
        train_dataset.shuffle()
        load_data_time = time() - start_load_data
        start_train_time = time()
        for i in tqdm(range(train_dataset.num_batch)):
            user_ids, item_ids, ratings = train_dataset.get_batch(i)
            loss_step, rmse_step = train_step(rs_model, optimizer, user_ids, item_ids, ratings)
            train_loss += loss_step
            train_rmse += rmse_step
            if i > 1000:
                break
        train_time = time() - start_train_time
        print('epoch: ', epoch, '. load data time: ', load_data_time,
              '. train time: ', train_time, '. train loss: ', train_loss.numpy(),
              '. train rmse: ', train_rmse.numpy() / (i * train_dataset.batch_size))
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
    args['mu'] = meta_data['mu']
    args['alpha'] = input_args.alpha
    args['beta'] = input_args.beta
    args['gamma'] = 0.000
    args['num_items'] = meta_data['num_items']
    args['num_users'] = meta_data['num_users']
    args['num_keywords'] = meta_data['num_keywords']
    args['item_keywords'] = item_keywords

    rsmodel = RSModel_7(args)
    opt = tf.keras.optimizers.Adam(learning_rate=input_args.lr)
    training(rsmodel, opt, train_dataset, val_dataset, num_epochs=input_args.epochs)


def predict(input_args):
    # load pretrained model
    train = pickle.load(open(model_folder + 'train.pkl', 'rb'))
    test = pickle.load(open(model_folder + 'test.pkl', 'rb'))

    meta_data = pickle.load(open(model_folder + 'meta_data.pkl', 'rb'))
    item_keywords = pickle.load(open(model_folder + 'item_keywords.pkl', 'rb'))

    args = dict()
    args['embedding_size'] = input_args.embed_size
    args['keyword_embedding_size'] = input_args.keyword_embed_size
    args['mu'] = meta_data['mu']
    args['alpha'] = input_args.alpha
    args['beta'] = input_args.beta
    args['gamma'] = 0.000
    args['num_items'] = meta_data['num_items']
    args['num_users'] = meta_data['num_users']
    args['num_keywords'] = meta_data['num_keywords']
    args['item_keywords'] = item_keywords

    # rec_model = RSModel(args)
    rec_model = RSModel_7(args)

    epoch_step = tf.Variable(0, dtype=tf.int32)
    ckpt = tf.train.Checkpoint(rec_model=rec_model, epoch_step=epoch_step)
    manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=model_folder + 'rsmodel_ckpt', max_to_keep=3)
    print('load pretrained model at: ' + manager.latest_checkpoint)
    ckpt.restore(manager.latest_checkpoint)

    test_dataset = DataSet(test[['userId', 'itemId', 'rating']].values, batch_size=1024)
    test_score = get_val_rmse(rec_model, test_dataset)
    print("test score: ", test_score)

    # prepare data for prediction
    item_ids = np.array(range(meta_data['num_items']))
    users_embedding = rec_model.user_embedding.weights[0].numpy()
    items_embedding = rec_model.item_embedding.weights[0].numpy()
    items_bias = np.squeeze(rec_model.bias_i.weights[0].numpy())
    users_bias = rec_model.bias_u.weights[0].numpy()

    # prepare item_keyword_encode
    item_keyword = tf.nn.embedding_lookup(rec_model.item_keywords, item_ids)
    item_keyword_embedding = rec_model.keyword_embedding(item_keyword)
    mask = rec_model.keyword_embedding.compute_mask(item_keyword)
    item_keyword_embedding = tf.multiply(item_keyword_embedding, tf.expand_dims(tf.cast(mask, tf.float32), 2))
    item_keyword_encode = tf.reduce_sum(item_keyword_embedding, axis=1)
    item_num_keywords = tf.reduce_sum(tf.cast(mask, tf.float32), axis=1) + 1e-8
    item_keyword_encode = tf.math.divide(item_keyword_encode, tf.expand_dims(item_num_keywords, 1))
    item_keyword_encode = rec_model.mlp_keyword_dense(item_keyword_encode)
    item_keyword_encode = np.squeeze(item_keyword_encode.numpy())

    # predict top_n for user
    predict_user_dict = dict()
    for user_id in tqdm(train['userId'].unique()):
        user_embedded = users_embedding[user_id]
        user_bias = users_bias[user_id]
        predicts = np.squeeze(np.matmul(user_embedded.reshape(1, -1), items_embedding.T)) + items_bias + user_bias + item_keyword_encode
        best = np.argpartition(predicts, -config.top_n)[-config.top_n:]
        predict_user_dict[user_id] = best.astype(np.int32)

    # save top_n user
    pickle.dump(predict_user_dict, open(model_folder + 'predict_user_dict.pkl', 'wb'))


def train_and_predict(input_args):
    train_model(input_args)
    predict(input_args)
    http_api_prepare.prepare()


if __name__ == '__main__':
    train_and_predict()
