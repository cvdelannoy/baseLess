import h5py
import numpy as np
from multiprocessing import Process, Queue

from baseLess.db_building.TrainingRead import Read
from baseLess.inference.ReadManager import ReadManager


def rolling_window(array, window_size,freq):
    shape = (array.shape[0] - window_size + 1, window_size)
    strides = (array.strides[0],) + array.strides
    rolled = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return rolled[np.arange(0,shape[0],freq)]


class ReadTable(object):

    def __init__(self, reads_dir, pos_reads_dir, predictor):
        """Issues new reads in correct format for Predictor, takes new predictions and passes them on to ReadManager
        process.

        :param reads_dir: Directory that contains fast5 reads on which to run inference
        :param pos_reads_dir: Directory where reads tested positive for sequence are stored
        :param predictor: object of Predictor class, read shapes are attuned to input requirements
        """
        self.pos_reads_dir = pos_reads_dir
        self.reads_dir = reads_dir
        self._batch_size = predictor.batch_size
        self._filter_width = predictor.filter_width
        self._stride = predictor.stride
        self._pred_queue = Queue()
        self._new_read_queue = Queue()


    def init_table(self):
        """Initialize table by starting up a manager process and return the process object"""
        manager_process = Process(target=ReadManager, args=(self._new_read_queue,
                                                            self._pred_queue,
                                                            self.reads_dir,
                                                            self.pos_reads_dir), name='read_manager')
        manager_process.start()
        return manager_process

    def get_reads(self, read_batch_size):
        """Get reads from ReadManager to predict, parse into format expected by Predictor.

        :param read_batch_size: integer of
        :return: Tuple of (path to read fast5 files,
                 Read object for inference (split to desired input length),
                 index list defining which rows in batch belong to which read,
                 list of unique read indices used in index list)
        """
        # try:
        #     read_fn = self._new_read_queue.get_nowait()  # stalls if queue is empty
        # except Empty:
        #     return None, None
        ii, fn_list = 0, []
        while ii < read_batch_size and not self._new_read_queue.empty():
            fn_list.append(self._new_read_queue.get_nowait())
            ii += 1
        if not len(fn_list): return None, None, None, None
        raw_list = []
        batch_idx_list = []
        batch_read_ids = []
        for ri, read_fn in enumerate(fn_list):
            with h5py.File(read_fn, 'r') as fh: raw = Read(fh, 'median').raw
            raw_strided = rolling_window(raw, self._filter_width, self._stride)
            batch_idx_list.extend([ri] * raw_strided.shape[0])
            batch_read_ids.append(ri)
            raw_list.append(raw_strided)
        nb_dummy_rows = self._batch_size - len(batch_idx_list) % self._batch_size
        raw_array = np.vstack(raw_list + [np.zeros(shape=(nb_dummy_rows, self._filter_width), dtype=raw_list[0].dtype)])
        batch_idx_list.extend([None] * nb_dummy_rows)
        return fn_list, raw_array, np.array(batch_idx_list), batch_read_ids

    def update_prediction(self, fn, pred):
        """Take fast5 file names and predictions, pass on to ReadManager to process"""
        for f, p in zip(fn, pred):
            self._pred_queue.put((f, p))
