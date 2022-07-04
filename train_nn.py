import random
import re

import yaml
import pickle
import importlib
import tensorflow as tf
import numpy as np

from bokeh.io import save, output_file
from os.path import basename, splitext
from datetime import datetime

import reader
from low_requirement_helper_functions import parse_output_path
from helper_functions import load_db, plot_timeseries


def train(parameter_file, training_data, test_data, plots_path=None,
          save_model=None, model_weights=None, quiet=False, tb_callback=None):
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    timestamp = datetime.now().strftime('%y-%m-%d_%H:%M:%S')
    # Load parameter file
    if type(parameter_file) == str:
        with open(parameter_file, 'r') as pf: params = yaml.load(pf, Loader=yaml.FullLoader)
    elif type(parameter_file) == dict:
        params = parameter_file
    else:
        raise ValueError(f'{type(parameter_file)} is not a valid data type for a parameter file')

    # Load train & test data
    test_db, ts_npzs = load_db(test_data, read_only=True)
    train_db, train_npzs = load_db(training_data, read_only=True)
    nb_examples = params['batch_size'] * params['num_batches']

    # Define path to additional graphs
    if plots_path:
        if plots_path[-1] != '/': plots_path += '/'
        plots_path = parse_output_path(plots_path + timestamp)
        sample_predictions_path = parse_output_path(plots_path+'sample_predictions/')
        ts_predict_path = parse_output_path(plots_path+'sample_npzs/')

    # Create save-file for model if required
    cp_callback = None
    if save_model:
        save_model_path = parse_output_path(save_model)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            save_model_path + timestamp,
            save_weights_only=True,
            save_freq=params['batch_size'])

    # create nn
    nn_class = importlib.import_module(f'nns.{params["nn_class"]}').NeuralNetwork
    nn = nn_class(**params, target=train_db.target, weights=model_weights,
                  cp_callback=cp_callback, tb_callback=tb_callback)

    # Start training
    worst_kmers = []
    x_val, y_val = test_db.get_training_set(nb_examples)
    for epoch_index in range(1, params['num_kmer_switches'] + 1):
        x_train, y_train = train_db.get_training_set(nb_examples, worst_kmers)  # todo worst_kmers mechanism errors out, fix
        nn.train(x_train, y_train, x_val, y_val,
                 eps_per_kmer_switch=params['eps_per_kmer_switch'], quiet=quiet)

        # if params['num_kmer_switches'] > 1:
        #     # Run on whole training reads for selection of top 5 wrongly classified k-mers
        #     random.shuffle(train_npzs)
        #     predicted_pos_list = []
        #     squiggle_count = 0
        #     for npz in train_npzs:
        #         x, sequence_length, kmers = reader.npz_to_tf_and_kmers(npz, params['max_sequence_length'], target_kmer=train_db.target)
        #         if x.shape[0] > nn.filter_width and np.any(np.in1d(train_db.target, kmers)):
        #             y_hat = nn.predict(x, clean_signal=True)
        #             predicted_pos_list.append(kmers[y_hat])
        #             squiggle_count += 1
        #             if squiggle_count == params['batch_size']:
        #                 break
        #     if not len(predicted_pos_list):
        #         predicted_pos = np.array([])
        #     else:
        #         predicted_pos = np.concatenate(predicted_pos_list)
        #     fp_kmers = Counter(predicted_pos)
        #     _ = fp_kmers.pop(train_db.target, None)
        #     nb_top = 5
        #     if len(fp_kmers) < nb_top:
        #         nb_top = len(fp_kmers)
        #     highest_error_rates = sorted(fp_kmers.values(), reverse=True)[:nb_top]
        #     worst_kmers = [k for k in fp_kmers if fp_kmers[k] in highest_error_rates]

        # Predict classification for a random trace and plot
        if plots_path:
            random.shuffle(ts_npzs)
            for i, npz in enumerate(ts_npzs):
                x, sequence_length, kmers = reader.npz_to_tf_and_kmers(npz, target_kmer=train_db.target)
                # x, sequence_length, kmers = reader.npz_to_tf_and_kmers(npz, params['max_sequence_length'], )
                if x.shape[0] > nn.filter_width and (np.any(np.in1d(train_db.target, kmers)) or i+1 == len(ts_npzs)):
                    tr_fn = splitext(basename(npz))[0]
                    start_idx = np.argwhere(np.in1d(kmers, train_db.target)).min()
                    oh = params['max_sequence_length'] // 2
                    sidx = np.arange(max(0,start_idx-oh), min(start_idx+oh, len(kmers)))
                    x = x[sidx, :]
                    kmers = kmers[sidx]
                    y_hat, posterior = nn.predict(x, clean_signal=True, return_probs=True)
                    ts_predict_name = ("{path}pred_ep{ep}_ex{ex}.npz".format(path=ts_predict_path,
                                                                             ep=epoch_index,
                                                                             ex=tr_fn))
                    if any(np.in1d(train_db.target, kmers)):
                        graph_start = np.min(np.where(np.in1d(kmers, train_db.target))) - 30
                    else:
                        graph_start = 0
                    ts_plot = plot_timeseries(raw=x,
                                              base_labels=kmers,
                                              posterior=posterior,
                                              y_hat=y_hat,
                                              start=graph_start,
                                              nb_classes=2)
                    output_file(sample_predictions_path + "pred_ep%s_ex%s.html" % (epoch_index, tr_fn))
                    reader.add_to_npz(npz, ts_predict_name, [y_hat, posterior, x, kmers, nn.target], ['labels_predicted', 'posterior', 'raw_excerpt', 'base_labels_excerpt', 'target'])
                    save(ts_plot)
                    break
                if (i+1) == len(ts_npzs):
                    raise UserWarning('None of test npzs suitable for testing, no squiggle plot generated.')

    # # Uncomment to print confusion matrix
    # # Rows are true labels, columns are predicted labels
    # prediction = nn.predict(x_val)
    # print(tf.math.confusion_matrix(y_val, prediction))
    return nn


def main(args):
    target = re.search('(?<=/)[^/]+/$', args.training_db).group(0)[:-1]
    nn_target_dir = parse_output_path(f'{args.nn_dir}{target}')
    tb_dir = parse_output_path(f'{nn_target_dir}tb_log/{datetime.now().strftime("%Y%m%d-%H%M%S")}')
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir, histogram_freq=1)
    nn = train(args.parameter_file, args.training_db, args.test_db, args.plots_path, args.ckpt_model,
               args.model_weights, False, tb_callback)
    nn.model.save(f'{nn_target_dir}nn.h5')
    with open(f'{nn_target_dir}performance.pkl', 'wb') as fh: pickle.dump(nn.history, fh)
