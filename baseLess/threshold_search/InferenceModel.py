import os

import tarfile
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from pathlib import Path
from os.path import splitext
from tempfile import TemporaryDirectory

from nns.keras_metrics_from_logits import precision, recall, binary_accuracy


def is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)

    prefix = os.path.commonprefix([abs_directory, abs_target])

    return prefix == abs_directory


def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")

    tar.extractall(path, members, numeric_owner=numeric_owner)

class InferenceModel(object):
    """Composite model of multiple k-mer recognising NNs."""

    def __init__(self, mod_fn, batch_size=32):
        """Construct model

        :param mod_fn: Path to tar file that contains tarred model output by compile_model.py
        :type mod_fn: str
        :param batch_size: Size of batch
        """
        self.load_model(mod_fn)
        self.batch_size = batch_size

    def predict(self, read, kmer):
        model_output = self._model_dict[kmer](read)
        return np.max(model_output)

    def load_model(self, mod_fn):
        """Load compiled model from path to saved precompiled model

        :param mod_fn: Path to tarred file of compiled model
        :return:
        """
        with TemporaryDirectory() as td:
            with tarfile.open(mod_fn) as fh:
                safe_extract(fh, td)
            out_dict = {}

            def loss_fun(y_true, y_pred):  # just dummy to satisfy the stupid thing being there
                msk = np.zeros(100, dtype=bool)
                msk[50] = True
                y_pred_single = tf.boolean_mask(y_pred, msk, axis=1)
                return K.binary_crossentropy(K.cast(y_true, K.floatx()), y_pred_single, from_logits=True)
            for mn in Path(td).iterdir():
                out_dict[splitext(mn.name)[0]] = tf.keras.models.load_model(mn,
                                                                            custom_objects={'precision': precision,
                                                                                            'recall': recall,
                                                                                            'binary_accuracy': binary_accuracy,
                                                                                            'loss_fun': loss_fun})
        # Dictionary with kmer string as key and keras.Sequential as value
        self._model_dict = out_dict
        # List of kmers that the InferenceModel contains models for
        self.kmers = list(self._model_dict)
        # Length of input that should be given to each individual model
        try:
            self.input_length = self._model_dict[list(self._model_dict)[0]].layers[0].input_shape[1]
        except:
            self.input_length = None  # todo for undefined timseries length nn, find better solution
