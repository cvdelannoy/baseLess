import argparse, subprocess

from os.path import basename, splitext
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

COMPLEMENT_DICT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}


def reverse_complement(km):
    return ''.join([COMPLEMENT_DICT[k] for k in km][::-1])


def fa2df(fa_fn, kmer_size, column_name, cores):
    """
    take multifasta file, return df of counts of  k-mers
    """
    with TemporaryDirectory() as tdo:
        subprocess.run(
            f'jellyfish count -t {cores} -m {kmer_size} -s {4 ** kmer_size} -C -o {tdo}/mer_counts.jf  {fa_fn}',
            shell=True)  # option C: no reverse complements
        kmer_dump = subprocess.run(f'jellyfish dump -c {tdo}/mer_counts.jf', shell=True, capture_output=True)
    kmer_freq_tuples = [km.split(' ') for km in kmer_dump.stdout.decode('utf-8').split('\n')]
    kmer_freq_tuples = [km for km in kmer_freq_tuples if len(km) == 2]
    kmer_freq_dict = {km[0]: int(km[1]) for km in kmer_freq_tuples}
    return pd.DataFrame.from_dict(kmer_freq_dict, orient='index', columns=[column_name])

def main(target_fasta, background_fastas, model_size, kmer_sizes, cores):
    nb_bg_fastas = len(background_fastas)
    kmers_per_bg = max(1, model_size // (nb_bg_fastas * 2))  # factor 2 because we need kmers from low and high end of ratio spectrum
    bg_fn_list = [splitext(basename(fn))[0] for fn in background_fastas]

    # --- parse fastas into dfs ---
    tdf_list = []
    bdf_meta_list = [[] for _ in range(nb_bg_fastas)]
    for kmer_size in kmer_sizes:
        tdf_list.append(fa2df(target_fasta, kmer_size, 'target', cores))
        for bi, (fn, bn_fn) in enumerate(zip(background_fastas, bg_fn_list)):
            bdf_meta_list[bi].append(fa2df(fn, kmer_size, bn_fn, cores))
    target_df = pd.concat(tdf_list)
    bg_df_list = [pd.concat(x) for x in bdf_meta_list]


    # --- select kmers based on relative abundances ---
    kmer_list = []
    count_df = pd.concat([target_df] + bg_df_list, axis=1)
    count_df.sort_values('target', ascending=False, inplace=True)
    sub_df_list = []
    count_df.loc[:, 'lt_quantile_target'] = count_df.loc[:, 'target'] < count_df.loc[:, 'target'].quantile(q=0.50)
    count_df.fillna(0, inplace=True)
    for bg_fn in bg_fn_list:
        count_df.loc[:, f'lt_quantile_{bg_fn}'] = count_df.loc[:, bg_fn] < count_df.loc[:, bg_fn].quantile(q=0.50)
        kmers_high = count_df.query(f'lt_quantile_{bg_fn}').iloc[:kmers_per_bg].index  # select k-mers high in target
        kmers_low = count_df.sort_values(bg_fn, ascending=False).query('lt_quantile_target').iloc[:kmers_per_bg].index  # select k-mers low in target
        cur_kmer_list = list(kmers_low) + list(kmers_high)
        kmer_bool = np.in1d(count_df.index, cur_kmer_list)
        sub_df_list.append(count_df.loc[kmer_bool, :])
        count_df = count_df.loc[~kmer_bool, :]  # remove selected k-mers so that next rounds do not select the same again
        kmer_list.extend(cur_kmer_list)
        # count_df.loc[:, f'ratio_{bg_fn}'] = count_df.target / count_df.loc[:, bg_fn]
        # ratio_series = count_df.loc[~np.in1d(count_df.index, kmer_list), f'ratio_{bg_fn}'].sort_values()
        # kmer_list.extend(ratio_series.iloc[-kmers_per_bg:].index)
        # kmer_list.extend(ratio_series.iloc[:kmers_per_bg].index)
    kmer_list = set(kmer_list)
    sub_df = pd.concat(sub_df_list)
    for cn in ['target'] + bg_fn_list:
        sub_df.loc[:, f'rel_{cn}'] = sub_df.loc[:, cn] / sub_df.loc[:, cn].sum()
    order_dict = {x: list(sub_df.sort_values(x).index) for x in ['target'] + bg_fn_list}
    return kmer_list, sub_df, order_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get set of k-mers for which abundance is maximally different.')
    # --- inputs ---
    parser.add_argument('--target-fasta', required=True)
    parser.add_argument('--background-fastas', nargs='+', required=True)
    # --- outputs ---
    parser.add_argument('--out-kmer-txt', required=True)
    parser.add_argument('--out-freq-table', required=False)
    # --- params ---
    parser.add_argument('--kmer-sizes', type=int, default=[9])
    parser.add_argument('--model-size', type=int, default=25)
    parser.add_argument('--cores', type=int, default=4)
    args = parser.parse_args()
    kmer_list, freq_df, order_dict = main(args.target_fasta, args.background_fastas, args.model_size, args.kmeer_sizes, args.cores)
    with open(args.out_kmer_txt, 'w') as fh: fh.write('\n'.join(kmer_list))
    if args.out_freq_table:
        freq_df.to_csv(args.out_freq_table)
