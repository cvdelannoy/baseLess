import h5py
import numpy as np
import tensorflow as tf
import ast

model_type_list = ['read_detection', 'abundance']

class Predictor(object):
    """predictor wrapper class for baseLess models"""

    def __init__(self, model_fn):
        if model_fn.endswith('.h5'):
            print('Loading model...')
            self.mod = tf.keras.models.load_model(model_fn)
            self.batch_size, self.filter_width, _ = self.mod.input.shape
            with h5py.File(model_fn, 'r') as fh:
                self.model_type = fh.attrs['model_type']
                self.kmer_list = fh.attrs['kmer_list'].split(',')
                self.nb_kmers = len(self.kmer_list)
                self.stride = fh.attrs['filter_stride']
                self.k_threshold = fh.attrs['k_threshold']
                if self.model_type == 'abundance':
                    order_dict_lists = ast.literal_eval(fh.attrs['order_dict'])
                    self.order_dict = {sp: np.array(order_dict_lists[sp]) for sp in order_dict_lists}
        else:
            raise ValueError('Non .h5 models deprecated')  # tflite and trt model implementations coming
        print(f'Done! Model type is {self.model_type}')

    @property
    def model_type(self):
        return self._model_type

    @model_type.setter
    def model_type(self, mt):
        if mt not in model_type_list:
            raise ValueError(f'Unsupported model type: {mt}. Check your baseLess version.')
        self._model_type = mt

    def predict(self, read, batch_idx_list, batch_read_ids):
        if self.model_type == 'read_detection':
            bin_list = []
            pred = []
            for bat in np.split(read, read.shape[0] // self.batch_size):
                pred_bin = self.mod(bat).numpy()
                bin_list.append(pred_bin)
            bin_array = np.vstack(bin_list)
            for brid in batch_read_ids:
                pred.append(np.sum(np.any(bin_array[batch_idx_list == brid], axis=0)) > self.k_threshold)
            return pred
        elif self.model_type == 'abundance':
            abundance_array = np.zeros(self.nb_kmers, dtype=float)
            for bat in np.split(read, read.shape[0] // self.batch_size):
                abundance_array += self.mod(bat).numpy()
            return abundance_array

    def get_msrd(self, abundance_estimate):
        order_estimated = [x for _, x in sorted(zip(abundance_estimate, self.kmer_list))]
        result_dict = {sp: np.mean([(np.argwhere(self.order_dict[sp] == kmer)[0, 0] - ki) ** 2
                               for ki, kmer in enumerate(order_estimated)]) for sp in self.order_dict}
        return result_dict
