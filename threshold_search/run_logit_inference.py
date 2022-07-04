import argparse, os, sys
from os.path import basename

import tensorflow as tf
import pandas as pd
from itertools import product
import h5py

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
baseless_location = os.path.realpath(f'{__location__}/..')
sys.path.append(baseless_location)

from low_requirement_helper_functions import parse_output_path, parse_input_path
from threshold_search.InferenceModel import InferenceModel
from db_building.TrainingRead import Read


class ReadTable(object):
    def __init__(self, reads_dir, input_length):
        self.input_length = input_length
        self.read_dict = {}
        reads_list = parse_input_path(reads_dir, pattern='*.fast5')
        self.read_fn_dict = {basename(fn): fn for fn in reads_list}
        self.read_fn_list = list(self.read_fn_dict)

    def load_read(self, fn, unsplit=False):
        if fn in self.read_dict: return self.read_dict[fn]
        with h5py.File(self.read_fn_dict[fn], 'r') as fh:
            if unsplit:
                self.read_dict[fn] = Read(fh, 'median').raw
            else:
                self.read_dict[fn] = Read(fh, 'median').get_split_raw_read(self.input_length)
        return self.read_dict[fn]


parser = argparse.ArgumentParser(description='Run inference for each k-mer on each read, return table of logits.')
parser.add_argument('--fast5-in', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--out-csv', type=str, required=True)
args = parser.parse_args()

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
mod = InferenceModel(args.model)

read_table = ReadTable(args.fast5_in, mod.input_length)
pred_df = pd.DataFrame(False, index=read_table.read_fn_list, columns=mod.kmers)
for fn, km in product(read_table.read_fn_list, mod.kmers):
    pred_df.loc[fn, km] = mod.predict(read_table.load_read(fn), km)

pred_df.to_csv(args.out_csv)
