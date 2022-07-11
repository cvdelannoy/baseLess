import argparse
import os
from baseLess.low_requirement_helper_functions import parse_output_path

# = ( , {
#     'type': ,
#     'required': ,
#     'help':
# })

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# --- Input ---
training_reads = ('--training-reads', {
    'type': str,
    'required': True,
    'help': 'Directory containing fast5 format reads, for nn training.'
})

test_reads = ('--test-reads', {
    'type': str,
    'required': True,
    'help': 'Directory containing fast5 format reads, for nn testing.'
})

training_db = ('--training-db', {
    'type': lambda x: check_db_input(x),
    'required': True,
    'help': 'Database generated by build_db, for training.'
})

test_db = ('--test-db', {
    'type': lambda x: check_db_input(x),
    'required': True,
    'help': 'Database generated by build_db, for testing.'
})

read_index = ('--read-index', {
    'type': str,
    'required': False,
    'help': 'Supply index file denoting which reads should be training reads and which should be test.'
})

kmer_list = ('--kmer-list', {
    'type': str,
    # 'required': True,
    'help': 'txt list of k-mers'
})

nn_directory = ('--nn-directory', {
    'type': str,
    'required': False,
    'default': f'{__location__ + "/data/16s_nns/nns/"}',
    'help': f'Directory containing single k-mer detecting neural networks from which baseLess classifiers are '
            f'constructed [default: {__location__ + "/data/16s_nns/nns/"}]'
})

model = ('--model', {
    'type': str,
    'required': True,
    'help': 'Combined k-mer model to use for inference.'
})

mem = ('--mem', {
    'type': int,
    'default': -1,
    'help': 'Amount of RAM to use in MB [default: no limit]'
    }
)

no_gpu = ('--no-gpu', {
    'action': 'store_true',
    'help': 'Do not use gpu [default: use gpu if available]'
})

batch_size = ('--batch-size', {
    'type': int,
    'default': 4,
    'help': 'Max number of reads for which to run prediction simultaneously, '
            'decreasing may resolve memory issues [default: 4]'
})

model_weights = ('--model-weights', {
    'type': str,
    'required': False,
    'default': None,
    'help': 'Provide a (tensorflow checkpoint) file containing graph meta data and weights '
            'for the selected model. '
})

parameter_file = ('--parameter-file', {
    'type': str,
    'required': False,
    'default': os.path.join(__location__, 'nns/hyperparams/CnnParameterFile.yaml'),
    'help': 'a yaml-file containing NN parameters. If none supplied, default values are used.'})

parameter_ranges_file = ('--parameter-ranges-file', {
    'type': str,
    'required': False,
    'default': os.path.join(__location__, 'hyperparameter_search/hyperparameter_ranges_cnn.yaml'),
    'help': 'a yaml-file containing valid ranges to search for optimal hyperparameters.'})

fast5_in = ('--fast5-in', {
    'type': lambda x: check_input_path(x),
    'required': True,
    'help': 'Folder containing fast5 reads'
})

# target_16S = ('--target-16S', {
#     'type': str,
#     'help': 'fasta containing to-be recognized 16S sequence(s)'
# })

inference_mode = ('--inference-mode', {
    'type': str,
    'choices': ['watch', 'once'],
    'default': 'watch',
    'help': 'Run inference [once] on fast5s available in folder, or [watch] folder indefinitely for new reads [default: watch]'
})

copy_reads = ('--copy-reads', {
    'action': 'store_true',
    'help': 'Copy reads to the output directory before starting processing'
})

# --- output ---
out_dir = ('--out-dir', {
    'type': lambda x: parse_output_path(x),
    'required': True
})

out_model = ('--out-model', {
    'type': str,
    'required': True,
    'help': 'Path to tar file in which to save produced model'
})

ckpt_model = ('--ckpt-model', {
    'type': str,
    'required': False,
    'default': None,
    'help': 'Store the model weights at the provided location, with the same name as the parameter (yaml) file '
            '(with ckpt-extension).'
})

nn_dir = ('--nn-dir', {
    'type': lambda x: parse_output_path(x),
    'required': True,
    'help': 'Directory where produced network(s) are stored. Networks get target k-mer as name.'
})

plots_path = ('--plots-path', {
    'type': str,
    'required': False,
    'default': None,
    'help': 'Define different location to store additional graphs, if made. Default is None (no graphs made) '
            'Folders and sub-folders are generated if not existiing.'
})

tensorboard_path = ('--tensorboard-path', {
    'type': str,
    'required': False,
    'default': os.path.expanduser('~/tensorboard_logs/'),
    'help': 'Define different location to store tensorboard files. Default is home/tensorboard_logs/. '
            'Folders and sub-folders are generated if not existiing.'
})

db_dir = ('--db-dir', {
    'type': str,
    'required': True,
    'help': 'Name (directory) new database'
})


# --- parameters ---

max_nb_examples = ('--max-nb-examples', {
    'type': int,
    'default': 10000,
    'help': 'Maximum number of examples to store in DB [default: 10000]'
})

nb_example_reads = ('--nb-example-reads', {
                       'type': int,
                        'default': 100,
                        'help': 'Number of full example reads to store [default: 100]'
})

silent = ('--silent', {
    'action': 'store_true',
    'help': 'Run without printing to console.'
})

cores = ('--cores', {
    'type': int,
    'default': 4,
    'help': 'Maximum number of CPU cores to engage at once.'
})

nb_folds = ('--nb-folds', {
    'type': int,
    'default': 5,
    'help': 'Number of cross validation folds [default: 5]'

})

target = ('--target', {
    'type': str,
    'required': True,
    'help': 'Target k-mer for db'
})

db_type = ('--db-type', {
    'type': str,
    'choices': ['train', 'test'],
    'default': 'train',
    'help': 'Denote whether the db is train or test. Only relevant if read index file is supplied [default: train]'
})

normalization = ('--normalization', {
    'type': str,
    'required': False,
    'default': 'median',
    'help': 'Specify how raw data should be normalized [default: median]'
})

width = ('--width', {
    'type': int,
    'required': True,
    'help': 'Filter width (Deprecated: get it from rnn params file)'
})

hdf_path = ('--hdf-path', {
    'type': str,
    'required': False,
    'default': 'Analyses/RawGenomeCorrected_000',
    'help': 'Internal path in fast5-files, at which analysis files can be found '
            '[default: Analyses/RawGenomeCorrected_000]'
})

uncenter_kmer = ('--uncenter-kmer', {
    'action': "store_true",
    'default': False,
    'help': 'If this flag is provided, kmers are not always centered in '
            'the read'
})

hyperopt_iters = ('--hyperopt-iters', {
    'type': int,
    'required': False,
    'default': 100,
    'help': 'Number of hyperopt hyperparameter optimization rounds [default: 100]'
})

hyperopt_parallel_jobs = ('--hyperopt-parallel-jobs', {
    'type': int,
    'required': False,
    'default': 4,
    'help': 'Number of hyperopt optimization to run in parallel [default: 4]'
})

# --- parser getters ---
def get_run_production_pipeline_parser():
    parser = argparse.ArgumentParser(description='Generate DBs from read sets and generate RNNs for several k-mers '
                                                 'at once')
    for arg in (training_reads, out_dir, kmer_list, cores,
                parameter_file, hdf_path, uncenter_kmer, read_index):
        parser.add_argument(arg[0], **arg[1])
    return parser


def get_build_db_parser():
    parser = argparse.ArgumentParser(description='Create ZODB database of training reads from resquiggled fast5s, '
                                                 'for given target k-mer')
    randomize = ('--randomize', {
        'action': 'store_true',
        'help': 'Randomize read order before adding examples'
    })
    for arg in (fast5_in, db_dir, normalization, target, width, hdf_path,
                uncenter_kmer, read_index, db_type, silent, nb_example_reads,
                max_nb_examples, randomize):
        parser.add_argument(arg[0], **arg[1])
    return parser


def get_training_parser():
    delete_dbs = ('--delete-dbs', {
        'action': 'store_true',
        'help': 'Delete train and test databases once training is done, to save space'
    })

    parser = argparse.ArgumentParser(description='Train a network to detect a given k-mer in MinION reads.')
    for arg in (training_db, test_db, nn_dir, tensorboard_path, plots_path, parameter_file,
                model_weights, ckpt_model, delete_dbs):
        parser.add_argument(arg[0], **arg[1])
    return parser


def get_run_inference_parser():
    parser = argparse.ArgumentParser(description='Start up inference routine and watch a fast5 directory for reads.')
    for arg in (fast5_in, out_dir, model, inference_mode, mem, no_gpu, batch_size, copy_reads):
        parser.add_argument(arg[0], **arg[1])
    # parser.add_argument('--continuous-nn', action='store_true',help='Used RNN can handle continuous reads.')
    return parser


def get_optimize_hyperparams_parser():
    parser = argparse.ArgumentParser(description='Find hyperparameters that minimize resource requirements while '
                                                 'keeping performance sufficiently high, on a provided list of k-mers')
    for arg in (training_reads, test_reads, kmer_list, parameter_ranges_file,
                out_dir, hyperopt_iters, hyperopt_parallel_jobs, cores, hdf_path, uncenter_kmer):
        parser.add_argument(arg[0], **arg[1])
    return parser


def get_compile_model_parser():
    parser = argparse.ArgumentParser(description='Compile a multi-network model from single k-mer models, for use '
                                                 'in run_inference.')

    model_type = ('--model-type', {
        'type': str,
        'default': 'read_detection',
        'choices': ['read_detection', 'abundance'],
        'help': 'Specify type of model to compile [read_detection] to determine identity of in each read, [abundance] for'
                'k-mer abundance estimation [default: read_detection]'
    })

    kmer_list = ('--kmer-list', {
        'type': str,
        'help': 'txt list of k-mers',
        'default': ''  # need it to be existent; passed to production pipeline
    })

    nb_kmers = ('--nb-kmers', {
        'type': int,
        'default': 25,
        'help': 'Number of kmers to detect in the classifier. higher==more specific but heavier on hardware [default: 25]'
    })

    accuracy_threshold = ('--accuracy-threshold', {
        'type': float,
        'required': False,
        'help': 'Only retain models with validation accuracy above given threshold [default: None]'
    })

    batch_size = ('--batch-size', {
        'type': int,
        'default': 32,
        'help': 'Number of read segments to simultaneously run inference on. Higher == potentially higher throughput but larger network [default: 32]'
    })

    target_fasta = ('--target-fasta', {
        'type': str,
        'help': 'Target sequence, full-length sequences for read detection mode or genome sequence for abundance mode'
    })

    background_fastas = ('--background-fastas', {
        'type': str,
        'nargs': '+',
        'help': '*ONLY USED IF --MODEL-TYPE abundance* background genomes, one fasta per genome'
    })

    for arg in (kmer_list, target_fasta, nn_directory, out_model, nb_kmers, parameter_file, batch_size, model_type,
                accuracy_threshold, background_fastas):
        parser.add_argument(arg[0], **arg[1])

    parser.add_argument('--train-required', action='store_true',
                        help='Train new models as required [default: use only available models]')

    training_reads = ('--training-reads', {
        'type': str,
        'required': False,
        'help': '*ONLY USED IF --TRAINING-REQUIRED* Directory containing fast5 format reads, for nn training.'
    })

    test_reads = ('--test-reads', {
        'type': str,
        'required': False,
        'help': '*ONLY USED IF --TRAINING-REQUIRED* Directory containing fast5 format reads, for nn testing.'
    })

    out_dir = ('--out-dir', {
        'type': lambda x: parse_output_path(x),
        'required': False,
        'help': '*ONLY USED IF --TRAINING-REQUIRED* nn output dir'
    })

    kmer_sizes = ('--kmer-sizes', {
        'type': int,
        'nargs': '+',
        'default': [8,9],
        'help': '*ONLY USED IF --TRAINING-REQUIRED* k-mer sizes of which networks can be generated [default: 8 9]'
    })

    cores = ('--cores', {
        'type': int,
        'default': 4,
        'help': '*ONLY USED IF --TRAINING-REQUIRED* Maximum number of CPU cores to engage at once.'
    })

    # parameter_file = ('--parameter-file', {
    #     'type': str,
    #     'required': False,
    #     'default': os.path.join(__location__, 'nns/hyperparams/RnnParameterFile_defaults.yaml'),
    #     'help': '*ONLY USED IF --TRAINING-REQUIRED* A yaml-file containing NN parameters. If none supplied, default values are used.'})

    hdf_path = ('--hdf-path', {
        'type': str,
        'required': False,
        'default': 'Analyses/RawGenomeCorrected_000',
        'help': '*ONLY USED IF --TRAINING-REQUIRED* Internal path in fast5-files, at which analysis files can be found '
                '[default: Analyses/RawGenomeCorrected_000]'
    })

    uncenter_kmer = ('--uncenter-kmer', {
        'action': "store_true",
        'default': False,
        'help': '*ONLY USED IF --TRAINING-REQUIRED* If this flag is provided, kmers are not always centered in '
                'the read'
    })

    read_index = ('--read-index', {
        'type': str,
        'required': False,
        'help': '*ONLY USED IF --TRAINING-REQUIRED* Supply index file denoting which reads should be training reads and which should be test.'
    })

    for arg in (training_reads, test_reads, out_dir, cores, hdf_path, uncenter_kmer, kmer_sizes, read_index):
        parser.add_argument(arg[0], **arg[1])
    return parser

# --- argument checking ---
def check_db_input(db_fn):
    """
    Check existence and structure, remove db.fs extension if necessary
    """
    if db_fn.endswith('db.fs'):
        if not os.path.isfile(db_fn):
            raise_(f'Database {db_fn} does not exist')
        return db_fn[:-5]
    elif not os.path.isdir(db_fn):
        raise_(f'Database {db_fn} does not exist')
    if db_fn[-1] != '/': db_fn += '/'
    if not os.path.isfile(f'{db_fn}db.fs'):
        raise_(f'{db_fn} not recognized as usable database (no db.fs found)')
    return db_fn


def check_input_path(fn):
    """
    Check if path exists, add last / if necessary
    """
    if not os.path.isdir(fn):
        raise_(f'Directory {fn} does not exist')
    if fn[-1] != '/': fn += '/'
    return fn


def raise_(ex):
    """
    Required to raise exceptions inside a lambda function
    """
    raise Exception(ex)



if __name__ == '__main__':
    raise ValueError('argument parser file, do not call.')
