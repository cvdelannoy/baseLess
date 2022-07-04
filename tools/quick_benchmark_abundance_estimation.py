import argparse, sys, os, shutil, yaml, re
import pandas as pd
from snakemake import snakemake
from glob import glob
from jinja2 import Template
from random import shuffle
from pathlib import Path

import multiprocessing as mp

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
baseless_location = os.path.realpath(f'{__location__}/..')
sys.path.append(baseless_location)
from low_requirement_helper_functions import parse_output_path
from tools.benchmark_helpers import plot_abundance_heatmap


def copy_dir(tup):
    read_list, out_species_dir, max_reads, rep = tup
    out_sample_dir = out_species_dir + f'{rep}/'
    shuffle(read_list)
    read_list = read_list[:max_reads]
    os.makedirs(out_sample_dir, exist_ok=True)
    for read_fn in read_list:
        shutil.copy(read_fn, out_sample_dir)


parser = argparse.ArgumentParser(description='Benchmark baseLess abundance estimation functionality')
# --- input ---
parser.add_argument('--index-csv', type=str, required=True,
                    help='Index file containing paths to genome and accompanying test reads directory.')
parser.add_argument('--max-test-reads', type=int, default=2000,
                    help='Maximum number of reads to sample from given test reads [default: 2000]')
parser.add_argument('--nb-repeats', type=int, default=5,
                    help='Number of repeated samplings to perform from given test reads [default: 5]')
parser.add_argument('--training-read-dir', type=str, required=True,
                    help='directory containing resquiggled training reads')
parser.add_argument('--kmer-mod-dir', type=str, required=False,
                    help='Directory containing pretrained nns')
#--- output ---
parser.add_argument('--out-dir', type=str, required=True)
#--- params ---
parser.add_argument('--parameter-file', type=str, default=f'{baseless_location}/nns/hyperparams/CnnParameterFile.yaml')
parser.add_argument('--model-size', type=int, default=25,
                    help='Define of how many k-mers abundance is estimated [default:25]')
parser.add_argument('--cores', type=int, default=4)
parser.add_argument('--min-kmer-mod-accuracy', type=float, default=0.85,
                    help='Filter kmer models on validation accuracy. May cause model size to shrink!')
parser.add_argument('--nb-gpus', type=int, default=1,
                    help='Number of GPUs that can be simultaneously engaged [default:1]')
parser.add_argument('--dryrun', action='store_true')

args = parser.parse_args()

out_dir = parse_output_path(args.out_dir, clean=True)
test_read_dir = parse_output_path(out_dir + 'test_reads')
genomes_dir = parse_output_path(out_dir + 'genomes')
analysis_dir = parse_output_path(out_dir + 'analysis')

species_dict = {}
sample_id_list = []
sample_dict = {}
mp_list = []
with open(args.index_csv, 'r') as fh:
    for line in fh.readlines():
        species, sample_id, genome_fn, test_dir = line.strip().split(',')
        if species in species_dict:
            assert genome_fn == species_dict[species]  # two samples for same species cannot list different genomes
        else:
            species_dict[species] = genome_fn
        if test_dir == 'None': continue
        sample_dict[species] = sample_dict.get(species, []) + [sample_id]

        # prepare copying test reads
        if test_dir[-1] != '/': test_dir += '/'
        sample_id_list.append(sample_id)
        read_list = [test_dir + fn for fn in os.listdir(test_dir)]
        chunk_size = len(read_list) // args.nb_repeats
        for ns in range(args.nb_repeats):
            mp_list.append((read_list[chunk_size * ns:chunk_size * (ns+1)], f'{test_read_dir}{species}/{sample_id}/', args.max_test_reads, ns))
with mp.Pool(min(len(sample_id_list), args.cores)) as pool:
    pool.map(copy_dir, mp_list)

for species in species_dict:
    os.symlink(species_dict[species], f'{genomes_dir}{species}.fasta')
    cur_bg_dir = parse_output_path(f'{genomes_dir}bg_{species}')
    for bgs in species_dict:
        if bgs == species: continue
        os.symlink(species_dict[bgs], f'{cur_bg_dir}{bgs}.fasta')

# kmer nns for different species need to be separated for simultaneous running. Pregenerate folders for that.
nn_dir = parse_output_path(out_dir + 'kmer_nns')
for species in list(sample_dict):
    _ = parse_output_path(nn_dir + species)
if args.kmer_mod_dir:
    kmer_mod_dir = args.kmer_mod_dir
    if kmer_mod_dir[-1] != '/': kmer_mod_dir += '/'
    kmer_mods_species = os.listdir(kmer_mod_dir)
    for species in list(sample_dict):
        if species in kmer_mods_species:
            shutil.copytree(f'{kmer_mod_dir}{species}/nns', f'{nn_dir}{species}/nns')

with open(args.parameter_file, 'r') as fh: param_dict = yaml.load(fh, Loader=yaml.FullLoader)

# --- generate snakemake file ---
with open(f'{__location__}/quick_benchmark_abundance_estimation.sf', 'r') as fh: template_txt = fh.read()
sf_txt = Template(template_txt).render(
    parameter_file=args.parameter_file,
    training_read_dir=args.training_read_dir,
    test_read_dir=test_read_dir,
    benchmark_dir=parse_output_path(out_dir + 'benchmarks'),
    inference_dir=parse_output_path(out_dir + 'inference'),
    kmer_list_dir=parse_output_path(out_dir + 'kmer_lists'),
    genomes_dir=genomes_dir,
    mod_dir=parse_output_path(out_dir + 'mods'),
    logs_dir=parse_output_path(out_dir + 'logs'),
    nn_dir=nn_dir,
    kmer_size=param_dict['kmer_size'],
    baseless_location=baseless_location,
    sample_dict=sample_dict,
    model_size=args.model_size,
    nb_repeats=args.nb_repeats,
    min_kmer_mod_accuracy=args.min_kmer_mod_accuracy
)

sf_fn = f'{out_dir}quick_benchmark_abundance_estimation.sf'
with open(sf_fn, 'w') as fh: fh.write(sf_txt)
resources = {}
if args.nb_gpus > 0:
    resources['gpu'] = args.nb_gpus
snakemake(sf_fn, cores=args.cores, keepgoing=True, dryrun=args.dryrun, resources=resources)
if args.dryrun: exit(0)


# Analyse results
msrd_csv_list = glob(f'{out_dir}inference/*/*/*/msrd.csv')
tup_list = []
for fn in msrd_csv_list:
    sample_id = Path(fn).parents[1].name
    species_id = Path(fn).parents[2].name.lstrip('inference_')
    msrd_df = pd.read_csv(fn)
    msrd_df.loc[msrd_df.query('species == "target"').index, 'species'] = species_id
    msrd_df.loc[:, 'sample_id'] = sample_id
    tup_list.append(msrd_df)
msrd_df = pd.concat(tup_list)
msrd_df.to_csv(f'{analysis_dir}msrd_all.csv')
plot_abundance_heatmap(msrd_df, analysis_dir)
