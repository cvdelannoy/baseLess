#!/usr/bin/python3
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from baseLess import train_nn, argparse_dicts, run_production_pipeline
from baseLess.hyperparameter_search import optimize_hyperparams
from baseLess.db_building import build_db
from baseLess.inference import run_inference
from baseLess.inference import compile_model
from baseLess.tools import update_16s_db

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    commands = [
        ('run_production_pipeline',
         argparse_dicts.get_run_production_pipeline_parser(),
         run_production_pipeline.main),
        ('train_nn',
         argparse_dicts.get_training_parser(),
         train_nn.main),
        ('hyperparameter_search',
         argparse_dicts.get_optimize_hyperparams_parser(),
         optimize_hyperparams.main),
        ('build_db',
         argparse_dicts.get_build_db_parser(),
         build_db.main),
        ('run_inference',
         argparse_dicts.get_run_inference_parser(),
         run_inference.main),
        ('compile_model',
         argparse_dicts.get_compile_model_parser(),
         compile_model.main),
        ('update_16s_db',
         argparse.ArgumentParser(description='Update database of standard neural networks. Pulled networks may be '
                                             'used for any purpose but are particularly useful for 16S detection.'),
         update_16s_db.main)
    ]

    parser = argparse.ArgumentParser(
        prog='baseLess',
        description='Build small neural networks to detect chunks of sequences in MinION reads.'
    )
    subparsers = parser.add_subparsers(
        title='commands'
    )

    for cmd, ap, fnc in commands:
        subparser = subparsers.add_parser(cmd, add_help=False, parents=[ap, ])
        subparser.set_defaults(func=fnc)
    args = parser.parse_args(args)
    args.func(args)


if __name__ == '__main__':
    main()
