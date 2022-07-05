import sys
import numpy as np
import pandas as pd
import h5py
from os.path import isdir, dirname, basename, splitext
from shutil import rmtree
from pathlib import Path
from random import shuffle

__location__ = dirname(Path(__file__).resolve())
sys.path.extend([__location__, f'{__location__}/..'])
from TrainingRead import TrainingRead
from ExampleDb import ExampleDb
from baseLess.low_requirement_helper_functions import parse_output_path, parse_input_path


def main(args):
    out_path = parse_output_path(args.db_dir)
    if isdir(out_path):
        rmtree(out_path)
    if args.read_index:
        read_index_df = pd.read_csv(args.read_index, index_col=0)
        if args.db_type == 'train':
            file_list = list(read_index_df.query(f'fold').fn)
        else:  # test
            file_list = list(read_index_df.query(f'fold == False').fn)
    else:
        file_list = parse_input_path(args.fast5_in, pattern='*.fast5')
    if args.randomize: shuffle(file_list)
    db_name = out_path+'db.fs'
    error_fn = out_path+'failed_reads.txt'
    npz_path = out_path + 'test_squiggles/'
    npz_path = parse_output_path(npz_path)
    kmer_size = len(args.target)

    db = ExampleDb(db_name=db_name, target=args.target, width=args.width)
    nb_files = len(file_list)
    count_pct_lim = 5
    nb_example_reads = 0
    for i, file in enumerate(file_list):
        try:
            with h5py.File(file, 'r') as f:
                tr = TrainingRead(f, normalization=args.normalization,
                                  hdf_path=args.hdf_path,
                                  kmer_size=kmer_size)
                nb_pos = db.add_training_read(training_read=tr,
                                              uncenter_kmer=args.uncenter_kmer)
            if nb_example_reads < args.nb_example_reads and nb_pos > 0:
                np.savez(npz_path + splitext(basename(file))[0], base_labels=tr.events, raw=tr.raw)
            if not i+1 % 10:  # Every 10 reads remove history of transactions ('pack' the database) to reduce size
                db.pack_db()
            if db.nb_pos > args.max_nb_examples:
                print('Max number of examples reached')
                break
            percentage_processed = int( (i+1) / nb_files * 100)
            if not args.silent and percentage_processed >= count_pct_lim:
                print(f'{percentage_processed}% of reads processed, {db.nb_pos} positives in DB')
                count_pct_lim += 5
        except (KeyError, ValueError) as e:
            with open(error_fn, 'a') as efn:
                efn.write('{fn}\t{err}\n'.format(err=e, fn=basename(file)))
            continue

    db.pack_db()
    if db.nb_pos == 0:
        raise ValueError(f'No positive examples found for kmer {args.target}')
