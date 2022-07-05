import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model

# from tensorflow.keras.metrics import Accuracy, Precision, Recall
import numpy as np
from helper_functions import clean_classifications

from nns.keras_metrics_from_logits import precision, recall, binary_accuracy

class NeuralNetwork(object):
    """
    Convolutional Neural network to predict target k-mer presence in squiggle

    :param target: The target k-mer that the NN should recognise
    :type target: str
    :param kernel_size: Kernel size of CNN
    :type kernel_size: int
    :param max_sequence_length: Maximum length of read that can be used to
    infer from. If read is longer than this length, don't use it.
    :type max_sequence_length: int
    :param weights: Path to h5 file that contains model weights, optional
    :param batch_size: Batch size to use during training
    :param threshold: Assign label to TRUE if probability above this threshold
                      when doing prediction
    :param eps_per_kmer_switch: Number of epochs to run
    :param filters: Number of CNN filters to use in the model
    :type filters: int
    :param learning_rate: Learning rate to use
    :param pool_size: If 0, do no pooling, else use this for the 1d maxpool
    """

    def __init__(self, **kwargs):
        self.target = kwargs['target']
        self.filter_width = kwargs['filter_width']
        self.hfw = (self.filter_width - 1) // 2  # half filter width
        self.kernel_size = kwargs['kernel_size']
        self.max_sequence_length = kwargs['max_sequence_length']
        self.batch_size = kwargs['batch_size']
        self.threshold = kwargs['threshold']
        self.eps_per_kmer_switch = kwargs['eps_per_kmer_switch']
        self.filters = kwargs['filters']
        self.learning_rate = kwargs['learning_rate']
        self.pool_size = kwargs['pool_size']
        self.dropout_remove_prob = 1 - kwargs['dropout_keep_prob']
        self.num_layers = kwargs['num_layers']
        self.batch_norm = kwargs['batch_norm']

        self.initialize(kwargs['weights'])
        self.history = {'loss': [], 'binary_accuracy': [], 'precision': [],
                        'recall': [], 'val_loss': [], 'val_binary_accuracy': [],
                        'val_precision': [], 'val_recall': []}

    def initialize(self, weights=None):
        """Initialize the network.

        :param weights: Path to .h5 model summary with weights, optional.
                        If provided, use this to set the model weights
        """
        if weights:
            self.model = tf.keras.models.load_model(weights, custom_objects={
                'precision': precision, 'recall': recall,
                'binary_accuracy': binary_accuracy})
            print('Successfully loaded weights')
            return

        # First layer
        self.model = models.Sequential()
        self.model.add(layers.Conv1D(self.filters,
                                     kernel_size=self.kernel_size,
                                     activation='relu',
                                     input_shape=(self.filter_width, 1)))
        for _ in range(self.num_layers):
            if self.batch_norm:
                self.model.add(layers.BatchNormalization())
            # self.model.add(layers.AvgPool1D(2))
            self.model.add(layers.Dropout(self.dropout_remove_prob))
            self.model.add(layers.Conv1D(self.filters,
                                         kernel_size=self.kernel_size,
                                         activation='relu'))
        if self.pool_size:
            self.model.add(layers.MaxPool1D(self.pool_size))
        # self.model.add(layers.GlobalMaxPool1D())
        self.model.add(layers.Flatten())
        self.model.add(layers.Dropout(self.dropout_remove_prob))
        self.model.add(layers.Dense(1, activation=None))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=[binary_accuracy, precision, recall])
        # if weights:
        #     self.model.load_weights(weights)
        #     print('Successfully loaded weights')

        # Uncomment to print model summary
        self.model.summary()

    def train(self, x, y, x_val, y_val, quiet=False, eps_per_kmer_switch=100):
        """Train the network. x_val/y_val may be used for validation/early
        stopping mechanisms.

        :param x: Input reads
        :param y: Ground truth labels of the read
        :param x_val: Input reads to use for validation
        :param y_val: Ground truth reads to use for validation
        :param quiet: If set to true, does not print to console
        """
        # Pad input sequences
        x_pad = np.expand_dims(pad_sequences(x, maxlen=self.filter_width,
                                             padding='post', truncating='post',
                                             dtype='float32'), -1)
        x_val_pad = np.expand_dims(pad_sequences(x_val,
                                                 maxlen=self.filter_width,
                                                 padding='post',
                                                 truncating='post',
                                                 dtype='float32'), -1)
        # Create tensorflow dataset
        tfd = tf.data.Dataset.from_tensor_slices((x_pad, y)).batch(
              self.batch_size).shuffle(x_pad.shape[0],
                                       reshuffle_each_iteration=True)

        # Train the model
        self.model.fit(tfd, epochs=self.eps_per_kmer_switch,
                       validation_data=(x_val_pad, y_val),
                       verbose=[2, 0][quiet])
        for metric in self.model.history.history:
            # If the metric at the final iteration is NaN, replace it with 1e-10
            # This makes sure the hyperparameter optimisation does not break
            if np.isnan(self.model.history.history[metric][-1]):
                self.model.history.history[metric][-1] = 1e-10
            self.history[metric].extend(self.model.history.history[metric])

    def predict(self, x, clean_signal=True, return_probs=False):
        """Given sequences input as x, predict if they contain target k-mer.
        Assumes the sequence x is a read that has been normalised,
        but not cut into smaller chunks.

        Function is mainly written to be called from train_nn.py.
        Not for final inference.

        :param x: Squiggle as numeric representation
        :type x: np.ndarray
        :param return_probs:
        :return: unnormalized predicted values
        :rtype: np.array of posteriors
        """
        offset = self.filter_width // 2
        ho = offset // 2
        lb, rb = self.hfw - ho, self.hfw + ho + 1
        idx = np.arange(self.filter_width, len(x) + offset, offset)
        x_batched = [x[si:ei] for si, ei in zip(idx-self.filter_width, idx)]

        x_pad = np.expand_dims(pad_sequences(x_batched, maxlen=self.filter_width,
                                             padding='post',
                                             truncating='post',
                                             dtype='float32'), -1)

        posteriors = self.model.predict(x_pad)
        y_hat = posteriors > self.threshold
        y_out = np.zeros(len(x), dtype=int)
        for i, yh in enumerate(y_hat):
            y_out[lb + i * offset: rb + i * offset] = yh
        if return_probs:
            posteriors_out = np.zeros(len(x), dtype=float)
            for i, p in enumerate(posteriors):
                posteriors_out[lb+i*offset: rb + i * offset] = p
            return y_out, posteriors_out
        return y_out