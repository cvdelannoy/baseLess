import argparse, os, sys, yaml, re
import pandas as pd
import numpy as np

from os.path import basename, dirname
from jinja2 import Template
from shutil import copy
from sklearn.model_selection import StratifiedShuffleSplit
from snakemake import snakemake


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
baseless_location = os.path.realpath(f'{__location__}/..')
sys.path.append(baseless_location)

from low_requirement_helper_functions import parse_output_path, parse_input_path
from tools.benchmark_helpers import parse_fast5_list


parser = argparse.ArgumentParser(description='Perform quick db+training+inference benchmark for k-mer set')
parser.add_argument('--fast5', type=str, required=True,
                    help='Fast5 directory')
kmer_source = parser.add_mutually_exclusive_group(required=True)
kmer_source.add_argument('--kmer-txt', type=str,
                    help='Txt file containing k-mers for which to run benchmark.')
kmer_source.add_argument('--target-16s', type=str,
                         help='fasta containing 16S sequences of target species.')
parser.add_argument('--nn-dir', type=str, required=True,
                    help='Directory in which kmer nns are stored.')
parser.add_argument('--ground-truth', type=str, required=True,
                    help='Ground truth text file.')
parser.add_argument('--parameter-file', type=str, required=True,
                    help='yaml defining network architecture.')
parser.add_argument('--max-test-size', type=int, default=100,
                    help='Test read set size [default: 100]')
parser.add_argument('--train-required', action='store_true')
parser.add_argument('--out-dir', type=str, required=True,
                    help='output directory.')
parser.add_argument('--hdf-path', type=str, default='Analyses/RawGenomeCorrected_000')
parser.add_argument('--cores', type=int, default=4,
                    help='Max number of cores engaged simultaneously [default: 4]')

args = parser.parse_args()

# --- parse inputs ---
fast5_list = np.array(parse_input_path(args.fast5, pattern='*.fast5'))
fast5_dir = dirname(fast5_list[0]) + '/'
if args.kmer_txt:
    with open(args.kmer_txt, 'r') as fh: kmer_list = [km.strip() for km in fh.readlines()]
elif args.target_16s:
    kmer_list = []
    with open(args.target_16s, 'r') as fh: fasta_txt = fh.read()
    # headers = [line for line in fasta_txt if line.startswith('>')]
    target_species = set(re.findall('(?<=>)[^,]+', fasta_txt))

with open(args.parameter_file, 'r') as pf: params = yaml.load(pf, Loader=yaml.FullLoader)

# --- generate folder structure ---
out_dir = parse_output_path(args.out_dir, clean=True)
logs_dir = parse_output_path(out_dir + 'logs')
nn_dir = out_dir + 'nns/'
db_dir = out_dir + 'dbs/'
inference_out_dir = out_dir + 'inference/'


# --- make read index file ---
gt_df = pd.read_csv(args.ground_truth, header=0, index_col='file_name')
if args.target_16s:
    gt_df.loc[:, 'pos'] = gt_df.species_short.apply(lambda x: x in target_species)
fast5_df = parse_fast5_list(fast5_list, gt_df)
gt_fn = f'{out_dir}ground_truth.csv'
gt_df.to_csv(gt_fn)

read_index_fn = f'{out_dir}read_index.csv'
train_num_idx, _ = list(StratifiedShuffleSplit(n_splits=1, test_size=0.1).split(fast5_df.index, fast5_df.pos))[0]
train_idx = fast5_df.index[train_num_idx]
fast5_df.loc[:, 'fold'] = False
fast5_df.loc[train_idx, f'fold'] = True
fast5_test_df = fast5_df.query('fold == False')
if len(fast5_test_df) > args.max_test_size:
    fast5_df = pd.concat((fast5_df.query('fold').copy(), fast5_test_df.sample(args.max_test_size).copy()))

fast5_df.to_csv(read_index_fn, columns=['fn', 'pos', 'fold'])

# --- make test reads folder ---
test_read_dir = parse_output_path(out_dir + 'test_reads')
for rc, (fn, tup) in enumerate(fast5_df.query('fold == False').iterrows()):
    copy(fast5_dir + fn, test_read_dir)

# --- render snakemake script ---
if args.kmer_txt:
    with open(f'{__location__}/quick_benchmark.sf', 'r') as fh: template_txt = fh.read()
    sm_text = Template(template_txt).render(
        baseless_location=baseless_location,
        read_dir=args.fast5,
        read_index_fn=read_index_fn,
        test_read_dir=test_read_dir,
        kmer_txt=args.kmer_txt,
        out_dir=out_dir,
        db_dir=db_dir,
        nn_dir=nn_dir,
        nn_target_list=kmer_list,
        inference_out_dir=inference_out_dir,
        logs_dir=logs_dir,
        parameter_file=args.parameter_file,
        filter_width=params['filter_width'],
        continuous_nn=params.get('continuous_nn', False),
        hdf_path=args.hdf_path
    )
elif args.target_16s and args.train_required:
    with open(f'{__location__}/quick_benchmark_16s.sf', 'r') as fh: template_txt = fh.read()
    sm_text = Template(template_txt).render(
        baseless_location=baseless_location,
        read_dir=args.fast5,
        read_index_fn=read_index_fn,
        ground_truth_fn=gt_fn,
        test_read_dir=test_read_dir,
        target_16s=args.target_16s,
        out_dir=out_dir,
        db_dir=db_dir,
        nn_dir=nn_dir,
        inference_out_dir=inference_out_dir,
        logs_dir=logs_dir,
        parameter_file=args.parameter_file,
        filter_width=params['filter_width'],
        hdf_path=args.hdf_path
    )
elif args.target_16s:
    with open(f'{__location__}/quick_benchmark_16s_noTrainRequired.sf', 'r') as fh: template_txt = fh.read()
    sm_text = Template(template_txt).render(
        baseless_location=baseless_location,
        read_dir=args.fast5,
        read_index_fn=read_index_fn,
        ground_truth_fn=gt_fn,
        test_read_dir=test_read_dir,
        target_16s=args.target_16s,
        out_dir=out_dir,
        nn_dir=args.nn_dir,
        inference_out_dir=inference_out_dir,
        logs_dir=logs_dir,
        parameter_file=args.parameter_file,
        filter_width=params['filter_width'],
        hdf_path=args.hdf_path
    )

sf_fn = f'{out_dir}quick_benchmark.sf'
with open(sf_fn, 'w') as fh: fh.write(sm_text)
snakemake(sf_fn, cores=args.cores, verbose=False, keepgoing=True, dryrun=False)
