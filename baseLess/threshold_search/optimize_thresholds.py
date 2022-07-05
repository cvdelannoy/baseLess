import sys, os, re, argparse
import pandas as pd
import snakemake as sm
import numpy as np
from pathlib import Path
from jinja2 import Template
from math import log

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__baseless__ = str(Path(f'{__location__}/..').resolve())
sys.path.append(__baseless__)

from low_requirement_helper_functions import parse_output_path


parser = argparse.ArgumentParser(description='Find logit and k-mer thresholds')
parser.add_argument('--fast5-train', type=str, required=True)
parser.add_argument('--fast5-test', type=str, required=True)
parser.add_argument('--ground-truth', type=str, required=True)
parser.add_argument('--targets-fasta', type=str, required=True)
parser.add_argument('--parameter-file', type=str, default=f'{__baseless__}/nns/hyperparams/CnnParameterFile.yaml')
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--logit-thresholds', type=float, nargs=3, default=[0.75, 0.999, 0.001])
parser.add_argument('--cores', type=int, default=4)
parser.add_argument('--dryrun', action='store_true')
args = parser.parse_args()

# make output folder structure
out_dir = parse_output_path(args.out_dir)
target_fastas_dir = parse_output_path(out_dir + 'target_fastas_dir')
model_dir = parse_output_path(out_dir + 'models')
logits_dir = parse_output_path(out_dir + 'logits')
thresholded_dir = parse_output_path(out_dir + 'thresholded')
histograms_dir = parse_output_path(out_dir + 'histograms')

logit_thresholds = [log(lt / (1 - lt)) for lt in np.arange(*args.logit_thresholds)]
species_list = list(pd.read_csv(args.ground_truth).species_short.unique())

with open(f'{__location__}/optimize_thresholds.sf', 'r') as fh:
    tempalate_txt = fh.read()
sf_txt = Template(tempalate_txt).render(
    # --- input ---
    all_targets_fasta=args.targets_fasta,
    fast5_train=args.fast5_train,
    fast5_test=args.fast5_test,
    ground_truth_csv=args.ground_truth,
    parameter_file=args.parameter_file,
    # --- output ---
    out_dir=out_dir,
    target_fastas_dir=target_fastas_dir,
    model_dir=model_dir,
    logits_dir=logits_dir,
    thresholded_dir=thresholded_dir,
    histograms_dir=histograms_dir,
    # --- params ---
    __baseless__=__baseless__,
    species_list=species_list,
    logit_threshold_list=logit_thresholds
)

sf_fn = f'{out_dir}optimize_thresholds.sf'
with open(sf_fn, 'w') as fh: fh.write(sf_txt)
sm.snakemake(sf_fn, cores=args.cores, keepgoing=True, dryrun=args.dryrun)

succeeded_species_list = [re.search('.+(?=_logits.csv)', fn).group(0) for fn in os.listdir(logits_dir)]

with open(f'{__location__}/optimize_thresholds_analysis.sf', 'r') as fh:
    tempalate_txt = fh.read()
sf_txt = Template(tempalate_txt).render(
    # --- input ---
    ground_truth_csv=args.ground_truth,
    model_dir=model_dir,
    # --- output ---
    out_dir=out_dir,
    logits_dir=logits_dir,
    thresholded_dir=thresholded_dir,
    histograms_dir=histograms_dir,
    # --- params ---
    __baseless__=__baseless__,
    species_list=succeeded_species_list,
    logit_threshold_list=logit_thresholds,
    dryrun=args.dryrun
)

sf_fn = f'{out_dir}optimize_thresholds_analysis.sf'
with open(sf_fn, 'w') as fh: fh.write(sf_txt)
sm.snakemake(sf_fn, cores=args.cores, keepgoing=True, dryrun=args.dryrun)
