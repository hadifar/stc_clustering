# -*- coding: utf-8 -*-

import os
from time import time

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.python.keras.optimizers import SGD

import metrics
from data_loader import load_data


def autoencoder(dims, act=tf.nn.leaky_relu, init='glorot_uniform'):
    n_stacks = len(dims) - 1
    # input
    x = tf.keras.layers.Input(shape=(dims[0],), name='input')
    h = x

    for i in range(n_stacks - 1):
        h = tf.keras.layers.Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)
    h = tf.keras.layers.Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)

    y = h
    for i in range(n_stacks - 1, 0, -1):
        y = tf.keras.layers.Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    y = tf.keras.layers.Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return tf.keras.models.Model(inputs=x, outputs=y, name='AE'), tf.keras.models.Model(inputs=x, outputs=h,
                                                                                        name='encoder')


class ClusteringLayer(tf.keras.layers.Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = tf.keras.layers.InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1].value
        self.input_spec = tf.keras.layers.InputSpec(dtype=tf.keras.backend.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform',
                                        name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (tf.keras.backend.sum(
            tf.keras.backend.square(tf.keras.backend.expand_dims(inputs, axis=1) - self.clusters),
            axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = tf.keras.backend.transpose(tf.keras.backend.transpose(q) / tf.keras.backend.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class STC(object):
    def __init__(self,
                 dims,
                 n_clusters=20,
                 alpha=1.0,
                 init='glorot_uniform'):

        super(STC, self).__init__()

        self.dims = dims
        self.input_dim = dims[0]
        self.n_stacks = len(self.dims) - 1

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.autoencoder, self.encoder = autoencoder(self.dims, init=init)

        # prepare DEC model
        clustering_layer = ClusteringLayer(self.n_clusters, name='clustering')(self.encoder.output)
        self.model = tf.keras.models.Model(inputs=self.encoder.input, outputs=clustering_layer)

    def pretrain(self, x, y=None, optimizer='adam', epochs=200, batch_size=256, save_dir='results/temp'):
        print('...Pretraining...')
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        if y is not None:
            class PrintACC(tf.keras.callbacks.Callback):
                def __init__(self, x, y):
                    self.x = x
                    self.y = y
                    super(PrintACC, self).__init__()

                def on_epoch_end(self, epoch, logs=None):
                    if int(epochs / 10) != 0 and epoch % int(epochs / 10) != 0:
                        return
                    feature_model = tf.keras.models.Model(self.model.input,
                                                          self.model.get_layer('encoder_3').output)
                    features = feature_model.predict(self.x)
                    km = KMeans(n_clusters=len(np.unique(self.y)), n_init=20, n_jobs=4)
                    y_pred = km.fit_predict(features)
                    # print()
                    print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
                          % (metrics.acc(self.y, y_pred), metrics.nmi(self.y, y_pred)))

        # begin pretraining
        t0 = time()
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs)
        print('Pretraining time: %ds' % round(time() - t0))
        self.autoencoder.save_weights(save_dir + '/ae_weights.h5')
        print('Pretrained weights are saved to %s/ae_weights.h5' % save_dir)
        self.pretrained = True

    def load_weights(self, weights):
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y=None, maxiter=2e4, batch_size=256, tol=1e-3,
            update_interval=140, save_dir='./results/temp', rand_seed=None):

        print('Update interval', update_interval)
        save_interval = int(x.shape[0] / batch_size) * 5  # 5 epochs
        print('Save interval', save_interval)

        # Step 1: initialize cluster centers using k-means
        print('Initializing cluster centers with k-means.')
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=100)
        y_pred = kmeans.fit_predict(self.encoder.predict(x))
        y_pred_last = np.copy(y_pred)
        self.model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        loss = 0
        index = 0
        index_array = np.arange(x.shape[0])
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q = self.model.predict(x, verbose=0)
                p = self.target_distribution(q)

                y_pred = q.argmax(1)
                if y is not None:
                    acc = np.round(metrics.acc(y, y_pred), 5)
                    nmi = np.round(metrics.nmi(y, y_pred), 5)
                    loss = np.round(loss, 5)
                    print('Iter %d: acc = %.5f, nmi = %.5f' % (ite, acc, nmi), ' ; loss=', loss)

                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            idx = index_array[index * batch_size: min((index + 1) * batch_size, x.shape[0])]
            loss = self.model.train_on_batch(x=x[idx], y=p[idx])
            index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

            ite += 1

        # save the trained model
        print('saving model to:', save_dir + 'STC_model_final.h5')
        self.model.save_weights(save_dir + 'STC_model_final.h5')

        return y_pred


if __name__ == "__main__":
    # args
    ####################################################################################
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='stackoverflow',
                        choices=['stackoverflow', 'biomedical', 'search_snippets'])

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--maxiter', default=1000, type=int)
    parser.add_argument('--pretrain_epochs', default=15, type=int)
    parser.add_argument('--update_interval', default=30, type=int)
    parser.add_argument('--tol', default=0.0001, type=float)
    parser.add_argument('--ae_weights', default='/data/search_snippets/results/ae_weights.h5')
    parser.add_argument('--save_dir', default='/data/search_snippets/results/')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.dataset == 'search_snippets':
        args.update_interval = 100
        args.maxiter = 100
    elif args.dataset == 'stackoverflow':
        args.update_interval = 500
        args.maxiter = 1500
        args.pretrain_epochs = 12
    elif args.dataset == 'biomedical':
        args.update_interval = 300
    else:
        raise Exception("Dataset not found!")

    print(args)

    # load dataset
    ####################################################################################
    x, y = load_data(args.dataset)
    n_clusters = len(np.unique(y))

    X_test, X_dev, y_test, y_dev = train_test_split(x, y, test_size=0.1, random_state=0)
    x, y = shuffle(X_test, y_test)

    # create model
    ####################################################################################
    dec = STC(dims=[x.shape[-1], 500, 500, 2000, 20], n_clusters=n_clusters)

    # pretrain model
    ####################################################################################
    if not os.path.exists(args.ae_weights):
        dec.pretrain(x=x, y=y, optimizer='adam',
                     epochs=args.pretrain_epochs, batch_size=args.batch_size,
                     save_dir=args.save_dir)
    else:
        dec.autoencoder.load_weights(args.ae_weights)

    dec.model.summary()
    t0 = time()
    dec.compile(SGD(0.1, 0.9), loss='kld')

    # clustering
    ####################################################################################
    y_pred = dec.fit(x, y=y, tol=args.tol, maxiter=args.maxiter, batch_size=args.batch_size,
                     update_interval=args.update_interval, save_dir=args.save_dir,
                     rand_seed=0)
    print('acc:', metrics.acc(y, y_pred))
    print('nmi', metrics.nmi(y, y_pred))
