#!/usr/bin/python3
import argparse
import sys

from baseLess import train_nn, argparse_dicts, run_production_pipeline
from baseLess.hyperparameter_search import optimize_hyperparams
from baseLess.db_building import build_db
from baseLess.inference import run_inference
from baseLess.inference import compile_model


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    commands = [
        ('run_production_pipeline',
         'Generate DBs from read set and generate NNs for several k-mers at once',
         argparse_dicts.get_run_production_pipeline_parser(),
         run_production_pipeline.main),
        ('train_nn',
         'Train a single NN to detect a given k-mer',
         argparse_dicts.get_training_parser(),
         train_nn.main),
        ('hyperparameter_search',
         'Find best combination of hyperparameters',
         argparse_dicts.get_optimize_hyperparams_parser(),
         optimize_hyperparams.main),
        ('build_db',
         'Build a training database, to train an NN for a given k-mer',
         argparse_dicts.get_build_db_parser(),
         build_db.main),
        ('run_inference',
         'Start up inference routine and watch a fast5 directory for reads.',
         argparse_dicts.get_run_inference_parser(),
         run_inference.main),
        ('compile_model',
         'Compile a multi-network model from single k-mer models, for use in run_inference.',
         argparse_dicts.get_compile_model_parser(),
         compile_model.main)
    ]

    parser = argparse.ArgumentParser(
        prog='baseLess',
        description='Build small neural networks to detect chunks of sequences in MinION reads.'
    )
    subparsers = parser.add_subparsers(
        title='commands'
    )

    for cmd, hlp, ap, fnc in commands:
        subparser = subparsers.add_parser(cmd, add_help=False, parents=[ap, ])
        subparser.set_defaults(func=fnc)

        # subparser = subparsers.add_parser(cmd, help=hlp, add_help=False, parents=[ap, ])
        # subparser.set_defaults(func=fnc)
    args = parser.parse_args(args)
    args.func(args)


if __name__ == '__main__':
    main()
