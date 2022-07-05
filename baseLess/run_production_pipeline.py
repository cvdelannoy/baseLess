import os
import yaml

from snakemake import snakemake
from jinja2 import Template

from baseLess.low_requirement_helper_functions import parse_output_path


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def main(args):

    db_dir = parse_output_path(f'{args.out_dir}dbs/')
    nn_dir = parse_output_path(f'{args.out_dir}nns/')
    logs_dir = parse_output_path(f'{args.out_dir}logs/')
    if type(args.kmer_list) == str:
        with open(args.kmer_list, 'r') as fh: kmer_list = [k.strip() for k in fh.readlines() if len(k.strip())]
    elif type(args.kmer_list) == list:
        kmer_list = args.kmer_list
    else:
        raise ValueError(f'dtype of kmer_list not valid: {type(args.kmer_list)}')
    with open(args.parameter_file, 'r') as pf: params = yaml.load(pf, Loader=yaml.FullLoader)

    # Construct and run snakemake pipeline
    with open(f'{__location__}/run_production_pipeline.sf', 'r') as fh: template_txt = fh.read()
    sm_text = Template(template_txt).render(
        __location__=__location__,
        db_dir=db_dir,
        nn_dir=nn_dir,
        logs_dir=logs_dir,
        parameter_file=args.parameter_file,
        train_reads=args.training_reads,
        test_reads=args.test_reads,
        read_index=args.read_index,
        read_index_bool=[True, False][args.read_index is None],
        kmer_list=kmer_list,
        filter_width=params['filter_width'],
        hdf_path=args.hdf_path,
        uncenter_kmer=args.uncenter_kmer
    )

    sf_fn = f'{args.out_dir}nn_production_pipeline.sf'
    with open(sf_fn, 'w') as fh: fh.write(sm_text)
    snakemake(sf_fn, cores=args.cores, verbose=False, keepgoing=True, resources={'gpu': 1})
