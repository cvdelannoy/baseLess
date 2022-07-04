import sys, os, re, yaml, subprocess, warnings, h5py, pickle

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from run_production_pipeline import main as rpp
from inference.diff_abundance_kmers import main as diff_abundance_kmers
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


COMPLEMENT_DICT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
kmer_freqs_df = pd.read_parquet(f'{__location__}/../data/ncbi_16S_bacteria_archaea_counts_8mer_9mer.parquet')
kmer_freqs_dict = kmer_freqs_df.kmer_count.to_dict()
with open(f'{__location__}/../data/list_of_kmers_after_19_rounds.csv', 'r') as fh:
    kmer_good_candidate_list = [km.strip() for km in fh.readlines() if len(km.strip())]


def reverse_complement(km):
    return ''.join([COMPLEMENT_DICT[k] for k in km][::-1])


def fa2kmers(fa_fn, kmer_size_list):
    """
    take multifasta file, return dict of lists of k-mers present in each sequence
    """
    with open(fa_fn, 'r') as fh:
        targets_txt = fh.read()
    targets_list = ['>' + t for t in targets_txt.split('>') if t]
    targets_dict = {re.search('(?<=>)[^\n]+', t).group(0): t for t in targets_list}
    kmer_dict = {}
    with TemporaryDirectory() as tdo:
        for td in targets_dict:
            kmer_dict[td] = []
            with open(tdo + '/target.fasta', 'w') as fh: fh.write(targets_dict[td])
            for kmer_size in kmer_size_list:
                kmer_dict[td].extend(count_kmers(tdo, kmer_size))
    return kmer_dict

def count_kmers(tdo, kmer_size):
    subprocess.run(
        f'jellyfish count -m {kmer_size} -s {4 ** kmer_size} -C -o {tdo}/mer_counts.jf  {tdo}/target.fasta',
        shell=True)
    kmer_dump = subprocess.run(f'jellyfish dump -c {tdo}/mer_counts.jf', shell=True, capture_output=True)
    kmer_list = [km.split(' ')[0] for km in kmer_dump.stdout.decode('utf-8').split('\n')]
    kmer_list = [km for km in kmer_list if len(km) == kmer_size]
    # kmer_list = [km for km in kmer_list if
    #              len(km) == kmer_size and km in kmer_good_candidate_list]  # retain k-mers found to score well in discernability  todo switched off for now
    kmer_list_curated = []  # remove reverse complements
    for km in kmer_list:
        if reverse_complement(km) in kmer_list_curated: continue
        kmer_list_curated.append(km)
    return kmer_list_curated

def get_kmer_candidates_16S(kmer_candidates_dict, min_nb_kmers=5, threshold=0.0001, filter_list=None):
    """
    get smallest set of 16S k-mers for which FDR<threshold
    """
    kmer_candidates_selection_dict = {}
    for kc in kmer_candidates_dict:
        if filter_list:
            prefiltered_kmer_list = [km for km in kmer_candidates_dict[kc] if km in filter_list]
        kmer_freqs_dict_cur = {km: kmer_freqs_dict.get(km, 0) for km in prefiltered_kmer_list}
        kmer_list = sorted(prefiltered_kmer_list, key=kmer_freqs_dict_cur.get, reverse=False)
        kmer_list = kmer_list[5:30]  # todo subset does not seem to have a positive effect
        kmer_candidates_selection_dict[kc] = kmer_list
        # kmer_candidates_selection_dict[kc] = []
        # sub_df = kmer_freqs_df.copy()
        # sub_freqs_dict = copy(kmer_freqs_dict)
        # cc=0
        # while cc <= len(kmer_list) and (p >= threshold or len(kmer_candidates_selection_dict[kc]) < min_nb_kmers):
        #     km = kmer_list.pop()  # pop least frequent k-mer
        #     kmer_candidates_selection_dict[kc].append(km)
        #
        #     # Select entries positive for k-mer or its reverse-complement
        #     km_rc = reverse_complement(km)
        #     hit_found = False
        #     if not len(sub_df):  # avoid making new queries if number of accessions is already 0, but min number of models has not been reached
        #         hit_found=True
        #     elif km in sub_df.columns and km_rc in sub_df.columns:
        #         sub_df = sub_df.query(f'{km} or {km_rc}').copy()
        #         hit_found=True
        #     elif km in sub_df.columns:
        #         sub_df.query(km, inplace=True)
        #         hit_found = True
        #     elif km_rc in sub_df.columns:
        #         sub_df.query(km_rc, inplace=True)
        #         hit_found = True
        #     if hit_found:
        #         hits = len(sub_df)
        #     else:
        #         hits = 1
        #     hits = max(hits, 1)
        #     p *= (hits - 1) / hits  # update FDR
        #     kmer_list.sort(key=sub_freqs_dict.get, reverse=True)  # re-sort kmer list
        #     sub_freqs_dict = dict(sub_df.sum(axis=0))
        #     cc += 1

        # if filter_list:
        #     kmer_candidates_selection_dict[kc] = [km for km in kmer_candidates_selection_dict[kc] if km in filter_list]
    kc_count_dict = {kc: len(kmer_candidates_selection_dict[kc]) for kc in kmer_candidates_selection_dict}
    selected_id = min(kc_count_dict, key=kc_count_dict.get)
    return kmer_candidates_selection_dict[selected_id]

def concat_conv1d_weights(weight_tensor, weight_list):
    weight_shapes = [w.shape for w in weight_list[0]]
    target_weight_shapes = [w.shape for w in weight_tensor]
    if weight_shapes[0][1] == target_weight_shapes[0][1]:
        return concat_weights(weight_list)
    nb_filters = weight_shapes[0][2]
    w_out = np.zeros(target_weight_shapes[0])
    b_out = []
    for i, w_sm in enumerate(weight_list):
        w_out[:, i*nb_filters:(i+1)*nb_filters, i*nb_filters:(i+1)*nb_filters] = np.array(w_sm[0])
        b_out.append(w_sm[1])
    return tf.convert_to_tensor(w_out), tf.concat(b_out, -1)

def concat_dense_weights(weight_tensor, weight_list):
    weight_shapes = [w.shape for w in weight_list[0]]
    target_weight_shapes = [w.shape for w in weight_tensor]
    if weight_shapes[0][0] == target_weight_shapes[0][0]:
        return concat_weights(weight_list)
    nb_dense = weight_shapes[0][0]
    w_out = np.zeros(target_weight_shapes[0])
    b_out = []
    for i, w_sm in enumerate(weight_list):
        w_out[i*nb_dense:(i+1)*nb_dense, i] = np.squeeze(np.array(w_sm[0]))
        b_out.append(w_sm[1])
    return tf.convert_to_tensor(w_out), tf.concat(b_out, -1)

def concat_weights(weight_list):
    concat_weight_list = []
    for iw in range(len(weight_list[0])):
        concat_weight_list.append(tf.concat([x[iw] for x in weight_list], -1))
    return concat_weight_list


def compile_model(kmer_dict, filter_width, threshold, batch_size, model_type):
    nb_mods = len(kmer_dict)
    input = tf.keras.Input(shape=(filter_width,1,), batch_size=batch_size)

    # construct and compile joined model
    trained_layers_dict = {}
    mod_first = tf.keras.models.load_model(f'{list(kmer_dict.values())[0]}/nn.h5', compile=False)
    x = input
    layer_index = 1
    for il, l in enumerate(mod_first.layers):
        if type(l) == tf.keras.layers.Dense:
            x = tf.keras.layers.Dense(l.weights[0].shape[1] * nb_mods, activation=l.activation)(x)
            trained_layers_dict[il] = {'ltype': 'dense', 'layer_index': layer_index, 'weights': []}
        elif type(l) == tf.keras.layers.Conv1D:
            x = tf.keras.layers.Conv1D(l.filters * nb_mods, l.kernel_size, activation=l.activation,
                                        groups=nb_mods if il > 0 else 1)(x)
            trained_layers_dict[il] = {'ltype': 'conv1d', 'layer_index': layer_index, 'weights': []}
        elif type(l) == tf.keras.layers.BatchNormalization:
            x = tf.keras.layers.BatchNormalization()(x)
            trained_layers_dict[il] = {'ltype': 'batchnormalization', 'layer_index': layer_index, 'weights': []}
        elif type(l) == tf.keras.layers.Dropout:
            x = tf.keras.layers.Dropout(l.rate)(x)
        elif type(l) == tf.keras.layers.MaxPool1D:
            x = tf.keras.layers.MaxPool1D(l.pool_size)(x)
        elif type(l) == tf.keras.layers.Flatten:
            nb_filters, t_dim = int(x.shape[2] / nb_mods), x.shape[1]
            x = tf.reshape(tf.expand_dims(x, -1), (batch_size, t_dim, nb_mods, nb_filters))
            x = tf.transpose(x, perm=[0, 2, 1, 3])
            x = tf.keras.layers.Flatten()(x)
            layer_index += 3
        else:
            raise ValueError(f'models with layer type {type(l)} cannot be concatenated yet')
        layer_index += 1
    output = K.cast_to_floatx(K.greater(x, threshold))
    meta_mod = tf.keras.Model(inputs=input, outputs=output)

    # collect weights
    for km in kmer_dict:
        mod = tf.keras.models.load_model(f'{kmer_dict[km]}/nn.h5', compile=False)
        for il, l in enumerate(mod.layers):
            if il not in trained_layers_dict: continue
            trained_layers_dict[il]['weights'].append(l.weights)

    # fill in weights
    for il in trained_layers_dict:
        if trained_layers_dict[il]['ltype'] == 'conv1d':
            weight_list = concat_conv1d_weights(meta_mod.layers[trained_layers_dict[il]['layer_index']].weights,
                                                trained_layers_dict[il]['weights'])
        elif trained_layers_dict[il]['ltype'] == 'dense':
            weight_list = concat_dense_weights(meta_mod.layers[trained_layers_dict[il]['layer_index']].weights,
                                               trained_layers_dict[il]['weights'])
        elif trained_layers_dict[il]['ltype'] == 'batchnormalization':
            weight_list = concat_weights(trained_layers_dict[il]['weights'])
        else:
            raise ValueError(f'Weight filling not implemented for layer type {trained_layers_dict[il]["ltype"]}')
        meta_mod.layers[trained_layers_dict[il]['layer_index']].set_weights(weight_list)
    if model_type == 'read_detection':
        meta_mod.compile()
        return meta_mod
    elif model_type == 'abundance':
        output = K.sum(meta_mod.output, axis=0)
        meta_mod_abundance = tf.keras.Model(inputs=meta_mod.input, outputs=output)
        meta_mod_abundance.compile()
        return meta_mod_abundance
    else:
        raise ValueError(f'--model-type {model_type} not implemented')

def train_on_the_fly(kmer_list, available_mod_dict, args):
    kmers_no_models = [km for km in kmer_list if km not in available_mod_dict]
    if len(kmers_no_models):  # train additional models, if required
        print(f'No models found for {len(kmers_no_models)} kmers, training on the fly!')
        args.kmer_list = kmers_no_models
        rpp(args)
        # add newly generated models to available model list
        for km in kmers_no_models:
            if os.path.exists(f'{args.out_dir}nns/{km}/nn.h5'):  # Check to filter out failed models
                available_mod_dict[km] = f'{args.out_dir}nns/{km}'
            else:
                warnings.warn(
                    f'model generation failed for {km}, see {args.out_dir}logs. Continuing compilation without it.')
                kmer_list.remove(km)
    out_dict = {km: available_mod_dict.get(km, None) for km in kmer_list if km in available_mod_dict}
    return out_dict


def filter_accuracy(kmer_dict, acc_threshold):
    out_dict = {}
    discard_list = []
    for kmd in kmer_dict:
        perf_fn = kmer_dict[kmd] + '/performance.pkl'
        if not os.path.isfile(perf_fn): continue
        with open(perf_fn, 'rb') as fh: perf_dict = pickle.load(fh)
        # metric = 2 / ( perf_dict['val_precision'][-1] ** -1 + perf_dict['val_recall'][-1] ** -1)  # F1
        metric = perf_dict['val_binary_accuracy'][-1]  # plain accuracy
        if metric > acc_threshold:
            out_dict[kmd] = kmer_dict[kmd]
        else:
            discard_list.append(kmd)
    return out_dict, discard_list


def main(args):

    # --- additional arg check ---
    if not args.kmer_list:
        if args.model_type == 'read_detection' and not args.target_fasta:
            raise ValueError('read_detection model_type requires providing a fasta containing target read sequences'
                             'with --target-fasta')
        elif args.model_type == 'abundance' and not (args.background_fastas and args.target_fasta):
            raise ValueError('Abundance mode requires providing fastas containing background genomes and the '
                             'target genome with --background-fastas and --target-fasta respectively.')

    # --- load nn parameters, list available kmer models ---
    with open(args.parameter_file, 'r') as fh:
        param_dict = yaml.load(fh, yaml.FullLoader)
    # List for which k-mers models are available
    if args.nn_directory:
        available_mod_dict = {pth.name: str(pth) for pth in Path(args.nn_directory).iterdir() if pth.is_dir()}
        kmer_size_list = list(np.unique([len(x) for x in available_mod_dict]))
    else:
        available_mod_dict = {pth.name: str(pth) for pth in Path(f'{__location__}/../data/16s_nns/').iterdir() if
                              pth.is_dir()}
        kmer_size_list = list(np.unique([len(x) for x in available_mod_dict]))
    if not len(kmer_size_list):
        kmer_size_list = [param_dict['kmer_size']]


    # --- Get target k-mers ---
    if args.kmer_list:  # parse a given list of kmers
        with open(args.kmer_list, 'r') as fh: requested_kmer_list = [km.strip() for km in fh.readlines()]
        target_kmer_list = [km for km in requested_kmer_list if len(km)]
    elif args.model_type == 'read_detection':  # estimate salient set of kmers from given 16S sequence
        requested_kmer_dict = fa2kmers(args.target_fasta, kmer_size_list)  # Collect k-mers per sequence in target fasta marked as recognizable todo check train_required implementation
        if args.train_required:
            target_kmer_list = get_kmer_candidates_16S(requested_kmer_dict, args.nb_kmers, 0.0001)
            target_kmer_dict = train_on_the_fly(target_kmer_list, available_mod_dict, args)
        else:  # filter out k-mers for which no stored model exists
            target_kmer_list = get_kmer_candidates_16S(requested_kmer_dict, args.nb_kmers, 0.0001, filter_list=list(available_mod_dict))
            target_kmer_dict = {km: available_mod_dict.get(km, None) for km in target_kmer_list if km in available_mod_dict}
        if not len(target_kmer_dict):
            raise ValueError('Sequences do not contain any of available models!')
    elif args.model_type == 'abundance': # for abundance mode
        target_kmer_list, abundance_freq_df, order_dict = diff_abundance_kmers(args.target_fasta, args.background_fastas,
                                                   args.nb_kmers, args.kmer_sizes, args.cores)
    else:
        raise ValueError('Either provide --kmer-list or --target-fasta')

    if args.train_required:
        target_kmer_dict = train_on_the_fly(target_kmer_list, available_mod_dict, args)
    else:
        target_kmer_dict = {km: available_mod_dict.get(km, None) for km in requested_kmer_list if
                            km in available_mod_dict}


    if args.accuracy_threshold:
        target_kmer_dict, discard_list = filter_accuracy(target_kmer_dict, args.accuracy_threshold)
        print(f'discarded {len(discard_list)} k-mers because accuracy is lower than threshold: {discard_list}')

    mod = compile_model(target_kmer_dict,
                        param_dict['filter_width'],
                        param_dict['threshold'],
                        args.batch_size,
                        args.model_type)
    mod.save(args.out_model)
    k_threshold = round(len(target_kmer_dict) * param_dict['k_frac'])
    with h5py.File(args.out_model, 'r+') as fh:
        fh.attrs['model_type'] = args.model_type
        fh.attrs['kmer_list'] = ','.join(list(target_kmer_dict))
        fh.attrs['k_threshold'] = k_threshold
        fh.attrs['filter_stride'] = param_dict['filter_stride']
        if args.model_type == 'abundance':
            fh.attrs['order_dict'] = str(order_dict)
    # if args.model_type == 'abundance':
    #     abundance_freq_df.to_hdf(args.out_model, 'abundance_freq_df')
