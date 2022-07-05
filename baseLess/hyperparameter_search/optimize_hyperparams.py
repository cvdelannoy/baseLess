import os, subprocess, yaml
import hyperopt as hp
from pathlib import Path
from random import getrandbits
from hyperopt.mongoexp import MongoTrials
from functools import partial
from datetime import datetime
from jinja2 import Template
from snakemake import snakemake

from baseLess.train_nn import train
from baseLess.low_requirement_helper_functions import parse_output_path

# # Uncomment for debugging:
# import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)


# Integer hyperparameters
integer_hps = ['layer_size', 'num_layers', 'filter_width', 'batch_size',
               'num_batches', 'num_kmer_switches', 'eps_per_kmer_switch',
               'kernel_size', 'pool_size', 'filters', 'num_layers',
               'batch_norm']

__location__ = os.path.dirname(Path(__file__).resolve())
baseless_location = Path(__file__).parents[1].resolve()

def generate_header(loss):
    out_txt = f"""
# BASELESS HYPEROPT-GENERATED PARAMETER FILE
# timestamp: {datetime.now().strftime('%y-%m-%d_%H:%M:%S')}
# loss: {loss}

"""
    return out_txt


def objective(params, non_variable_args, max_nb_params, paths):
    loss_list = []
    for (training_data, test_data) in paths:
        for nv in non_variable_args:
            params[nv] = non_variable_args[nv]
        for ih in integer_hps:
            if ih in params:
                params[ih] = int(params[ih])
        nn = train(parameter_file=params, training_data=training_data,
                   test_data=test_data, quiet=True)

        # Added this to prevent division by 0 errors:
        precision = nn.history['val_precision'][-1]
        recall = nn.history['val_recall'][-1]
        if not precision > 0.0:
            print('Rounding up precision')
            precision = 1e-10
        if not recall > 0.0:
            print('Rounding up recall')
            recall = 1e-10
        f1 = 2 / (precision ** -1 + recall ** -1)
        loss = (1 - f1) + nn.model.count_params() / max_nb_params * 0.01
        loss_list.append(loss)
    mean_loss = sum(loss_list) / len(loss_list)
    return {'loss': mean_loss, 'status': hp.STATUS_OK}


def main(args):

    os.environ["PYTHONPATH"] = str(baseless_location) + os.pathsep \
                               + os.environ.get('PYTHONPATH', '')

    # load hyperparameter ranges
    with open(args.parameter_ranges_file, 'r') as fh:
        ranges_dict = yaml.load(fh, yaml.FullLoader)

    # Generate dbs
    with open(args.kmer_list, 'r') as fh:
        kmer_list = [k.strip() for k in fh.readlines() if len(k.strip())]

    db_dir = parse_output_path(f'{args.out_dir}dbs/')
    logs_dir = parse_output_path(f'{args.out_dir}logs/')

    with open(f'{__location__}/generate_dbs.sf', 'r') as fh: template_txt = fh.read()
    sm_text = Template(template_txt).render(
        __location__=baseless_location,
        db_dir=db_dir,
        logs_dir=logs_dir,
        train_reads=args.training_reads,
        test_reads=args.test_reads,
        kmer_list=kmer_list,
        filter_width=ranges_dict['nonvariable']['filter_width'],
        hdf_path=args.hdf_path,
        uncenter_kmer=args.uncenter_kmer
    )
    sf_fn = f'{args.out_dir}db_production_pipeline.sf'
    with open(sf_fn, 'w') as fh:
        fh.write(sm_text)
    snakemake(sf_fn, cores=args.cores, verbose=False, keepgoing=True)

    paths = [(f'{db_dir}train/{km}', f'{db_dir}test/{km}') for km in kmer_list]

    # Define search space
    space = {}
    for var in ranges_dict['variable']:
        cd = ranges_dict['variable'][var]
        space[var] = hp.hp.quniform(var, cd['min'], cd['max'], cd['step'])

    # formulate objective function
    fmin_objective = partial(objective, non_variable_args=ranges_dict['nonvariable'],
                             max_nb_params=ranges_dict['max_nb_params'], paths=paths)

    # Start mongod process
    mongodb_pth = parse_output_path(f'{args.out_dir}mongodb', clean=True)
    subprocess.run(["mongod", "--dbpath", mongodb_pth, "--port", "1234",
                    "--directoryperdb", "--fork", "--journal",
                    "--logpath", f"{args.out_dir}mongodb_log.log"])


    # start worker processses
    worker_cmd_list = ["hyperopt-mongo-worker", "--mongo=localhost:1234/db"]
    worker_list = [subprocess.Popen(worker_cmd_list,
                                    stdout=open(os.devnull, 'wb'),
                                    stderr=open(os.devnull, 'wb'))
                   for _ in range(args.hyperopt_parallel_jobs)]

    # Minimize objective
    trials = MongoTrials('mongo://localhost:1234/db/jobs', exp_key=str(getrandbits(32)))
    out_param_dict = hp.fmin(fmin_objective,
                             space=space,
                             algo=hp.tpe.suggest,
                             trials=trials,
                             max_evals=args.hyperopt_iters,
                             max_queue_len=10)
    for worker in worker_list:
        worker.terminate()
    subprocess.run(['mongod', '--shutdown', '--dbpath', mongodb_pth])

    print(out_param_dict)
    for p in out_param_dict:
        if ranges_dict['variable'][p]['type'] == 'int':
            out_param_dict[p] = int(out_param_dict[p])
    for p in ranges_dict['nonvariable']:
        out_param_dict[p] = ranges_dict['nonvariable'][p]
    # store hyperparams as new parameter file
    with open(f'{args.out_dir}parameterFile.yaml', 'w') as fh:
        fh.write(generate_header(trials.losses()[-1]))
        yaml.dump(out_param_dict, fh)
