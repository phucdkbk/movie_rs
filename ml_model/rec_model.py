import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import TruncatedNormal


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
        self.mlp_keyword_dense = tf.keras.layers.Dense(units=1, activation='tanh')

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
        item_encode = self.mlp_keyword_dense(item_encode)
        # rating score
        r = tf.squeeze(user_bias) + tf.squeeze(item_bias) + tf.reduce_sum(mf, axis=1) + tf.reduce_sum(item_encode, axis=1)

        #         r = tf.squeeze(user_bias) + tf.squeeze(item_bias) + tf.reduce_sum(mf, axis=1)
        return r

    def loss_fn_rmse(self, predictions, labels):
        rmse = tf.reduce_sum(tf.math.square(predictions - labels))
        loss = rmse + tf.reduce_sum(self.keyword_embedding.losses)
        loss += tf.reduce_sum(self.user_embedding.losses) + tf.reduce_sum(self.item_embedding.losses)
        #         loss += tf.reduce_sum(self.bias_u.losses) + tf.reduce_sum(self.bias_i.losses)
        return loss, rmse

'''
    - keyword mask
    - add global rating meam mu
'''
class RSModel_7(Model):

    def __init__(self, args):
        super(RSModel_7, self).__init__()
        self.embedding_size = args['embedding_size']
        self.keyword_embedding_size = args['keyword_embedding_size']
        self.mu = args['mu']
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
        self.mlp_keyword_dense = tf.keras.layers.Dense(units=1)

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
        mask = self.keyword_embedding.compute_mask(item_keyword)
        item_keyword_embedding = tf.multiply(item_keyword_embedding, tf.expand_dims(tf.cast(mask, tf.float32), 2))
        item_encode = tf.reduce_sum(item_keyword_embedding, axis=1)
        item_num_keywords = tf.reduce_sum(tf.cast(mask, tf.float32), axis=1) + 1e-8
        item_encode = tf.math.divide(item_encode, tf.expand_dims(item_num_keywords, 1))

        item_encode = self.mlp_keyword_dense(item_encode)
        item_encode = tf.reduce_sum(item_encode, axis=1)

        # rating score
        r = self.mu + tf.squeeze(user_bias) + tf.squeeze(item_bias) + tf.reduce_sum(mf, axis=1) + item_encode
        return r

    def loss_fn_rmse(self, predictions, labels):
        rmse = tf.reduce_sum(tf.math.square(predictions - labels))
        loss = rmse + tf.reduce_sum(self.keyword_embedding.losses)
        loss += tf.reduce_sum(self.user_embedding.losses) + tf.reduce_sum(self.item_embedding.losses)
        loss += tf.reduce_sum(self.mlp_keyword_dense.losses)
        return loss, rmse