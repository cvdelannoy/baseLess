import argparse, re, os, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import chain
from os.path import basename
from collections.abc import Iterable

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
baseless_location = os.path.realpath(f'{__location__}/..')
sys.path.append(baseless_location)

from low_requirement_helper_functions import parse_output_path


def parse_fast5_list(fast5_list, gt_df):
    fast5_basename_list = [basename(f) for f in fast5_list]
    fast5_df = pd.DataFrame({'read_id': fast5_basename_list,
                             'fn': fast5_list,
                             'pos': [gt_df.pos.get(ff, 'unknown') for ff in fast5_basename_list]}
                            ).set_index('read_id')

    fast5_df.drop(fast5_df.query('pos == "unknown"').index, axis=0, inplace=True)
    fast5_df.pos = fast5_df.pos.astype(bool)
    return fast5_df


def format_species_names(x):
    element_list = x.split('_')
    genus, species = element_list[0], element_list[1]
    if len(genus) == 1: genus = genus.upper() + '.'
    out_name = f'$\it{{{genus} {species}}}$'
    return out_name


def get_freqtable(ft_fn, kmers):
    ft_df = pd.read_csv(ft_fn, index_col=0).loc[kmers, :]
    ft_df = ft_df[~ft_df.index.duplicated(keep='first')]
    return ft_df


def get_rank_score(s1, s2):
    sp1_list = np.array(s1.sort_values().index)
    sp2_list = np.array(s2.sort_values().index)
    # half_point = len(sp1_list) // 2
    # lower_half = np.sum(np.in1d(sp1_list[:half_point], sp2_list[:half_point]))
    # upper_half = np.sum(np.in1d(sp1_list[half_point:], sp2_list[half_point:]))
    # rank_score = (lower_half + upper_half) / len(sp1_list)
    rs_list = [(np.argwhere(sp2_list == kmer)[0, 0] - ki) ** 2 for ki, kmer in enumerate(sp1_list)]

    rs_list2 = []
    for si1 in s1.index:
        r1, r2 = np.argwhere(sp1_list == si1)[0,0], np.argwhere(sp2_list == si1)[0,0]
        rs = np.sqrt((r1 - r2) ** 2)
        rs_list2.append(rs)
    # return rank_score, rs_list2
    return np.mean(rs_list), rs_list2


def analyse_abundance_results(freq_tables_dict, analysis_sample_dict, all_species_list, out_dir):
    out_dir = parse_output_path(out_dir)

    timeseries_list = []
    # test_species_list = list(analysis_sample_dict)
    sample_list = list(chain.from_iterable([list(analysis_sample_dict[sp]) for sp in analysis_sample_dict]))
    nb_samples = len(sample_list)
    rankscore_df = pd.DataFrame(columns=['sample_id', 'rep_nb'] + all_species_list).set_index(['sample_id', 'rep_nb'])
    for species in analysis_sample_dict:
        for sa in analysis_sample_dict[species]:
            for rep in analysis_sample_dict[species][sa]:
                ae_df = pd.read_csv(analysis_sample_dict[species][sa][rep], index_col=0)
                kmers = list(ae_df.columns)
                kmer_rankscore_df_list = []
                ae_cumsum_df = ae_df.cumsum()
                ts_df = pd.DataFrame({'species': species, 'sample_id': sa, 'rep': rep, 'rank_score': None}, index=ae_cumsum_df.index)

                ft_df = get_freqtable(freq_tables_dict[species], kmers)
                for bg_sp in all_species_list:
                    col_idx = 'target' if bg_sp == species else bg_sp
                    rank_scores = ae_cumsum_df.apply(lambda x: get_rank_score(x, ft_df.loc[:, col_idx]), axis=1).apply(pd.Series)
                    ts_df.loc[:, bg_sp] = rank_scores[0]
                    kmer_rankscore_df_list.append(pd.DataFrame({'kmer': ae_cumsum_df.columns, 'species': bg_sp, 'mrd': rank_scores[1].iloc[-1]}))
                    rankscore_df.loc[(sa, rep), bg_sp] = ts_df.iloc[-1].loc[bg_sp]
                timeseries_list.append(ts_df)

                kmer_rankscore_df = pd.concat(kmer_rankscore_df_list)
                kmer_rankscore_df.loc[:, 'target_abundance'] = kmer_rankscore_df.kmer.apply(lambda x: ft_df.loc[x, 'target'])
                kmer_rankscore_df.sort_values('target_abundance', inplace=True)
                cols = [col for col in ft_df.columns if col.startswith('rel_')]
                ft_plot_df = ft_df.reset_index().loc[:, cols + ['index']].melt(value_vars=cols, id_vars='index')
                ft_plot_df.loc[:, 'target_abundance'] = ft_plot_df.loc[:, 'index'].apply(lambda x: ft_df.loc[x, 'target'])
                ft_plot_df.sort_values('target_abundance', inplace=True)

                fig, ax = plt.subplots(2,1, figsize=(10,10))
                sns.barplot(y='mrd', x='kmer', hue='species', data=kmer_rankscore_df, ax=ax[0])
                sns.stripplot(x='index', y='value', hue='variable', data=ft_plot_df, ax=ax[1])

                plt.setp(ax[0].get_xticklabels(), rotation=-90)
                plt.setp(ax[1].get_xticklabels(), rotation=-90)
                plt.tight_layout()
                plt.savefig(f'{out_dir}kmer_contributions_{sa}_{rep}.svg')
                plt.close(fig)


    rankscore_df.to_csv(f'{out_dir}rds_final.csv')
    timeseries_df = pd.concat(timeseries_list)
    timeseries_df.reset_index(inplace=True)
    timeseries_df.rename({'index': 'nb_reads'}, inplace=True, axis=1)
    timeseries_df.to_csv(f'{out_dir}rds_timeseries.csv')

    # --- heatmap ---
    heat_df = pd.DataFrame(index=sample_list, columns=all_species_list, dtype="float")
    heat_df_sd = heat_df.copy()
    for sp_true in analysis_sample_dict:
        for sid in analysis_sample_dict[sp_true]:
            for sp_pred in all_species_list:
                heat_df.loc[sid, sp_pred] = rankscore_df.loc[(sid, ), sp_pred].mean()
                heat_df_sd.loc[sid, sp_pred] = rankscore_df.loc[(sid,), sp_pred].std()
    # heat_df = heat_df.div(heat_df.sum(axis=1), axis=0)  # row-normalization
    annot_mat = (heat_df.round(2).astype(str) + '\n±' + heat_df_sd.round(2).astype(str))

    # heat_df.index = [format_species_names(x) for x in heat_df.index]
    heat_df.columns = [format_species_names(x) for x in heat_df.columns]
    heat_df.rename_axis('Truth', axis='rows', inplace=True)
    heat_df.rename_axis('Predicted', axis='columns', inplace=True)
    heat_df.to_csv(f'{out_dir}heat_df.csv')
    annot_mat.to_csv(f'{out_dir}heat_df_annot.csv')

    for cmap in ('Blues', 'Oranges', 'Greens'):
        fig, ax = plt.subplots(figsize=(5,5))
        sns.heatmap(data=heat_df, annot=annot_mat, fmt='s', cmap=cmap)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(title='RDS')
        plt.savefig(f'{out_dir}heatmap_rankdiff_{cmap}.svg', dpi=400)
        plt.close(fig)

    # --- timeseries plot ---
    fig, axes = plt.subplots(1,nb_samples, figsize=(18,5))
    if not isinstance(axes, Iterable): axes = [axes]
    mid_plot_idx = nb_samples // 2
    for si, sa in enumerate(sample_list):
        tsmelt_df = timeseries_df.query(f'sample_id == "{sa}"').melt(id_vars='nb_reads', value_vars=all_species_list).rename({'value': 'RDS', 'variable': 'species', 'nb_reads': '# reads'}, axis=1)
        tsmelt_df.species = tsmelt_df.species.apply(lambda x: format_species_names(x))
        sns.lineplot(x='# reads', y='RDS', hue='species', data=tsmelt_df, ax=axes[si])
        axes[si].set_title(sa)
        if si != nb_samples - 1: axes[si].legend().remove()
        if si != 0: axes[si].set_ylabel('')
        if si != mid_plot_idx: axes[si].set_xlabel('')
    plt.savefig(f'{out_dir}timeseries_rankdiff.svg', dpi=400)

def plot_abundance_heatmap(msrd_df, out_dir):
    confusion_dict = {}
    confusion_dict_sd = {}
    for sample_id, sdf in msrd_df.groupby('sample_id'):
        sub_dict = {}
        confusion_dict[sample_id] = sdf.groupby('species').msrd.mean()
        confusion_dict_sd[sample_id] = sdf.groupby('species').msrd.std()
    heat_df = pd.concat(confusion_dict, axis=1).T
    heat_df_sd = pd.concat(confusion_dict_sd, axis=1).T
    annot_mat = (heat_df.round(2).astype(str) + '\n±' + heat_df_sd.round(2).astype(str))
    heat_df.columns = [format_species_names(x) for x in heat_df.columns]
    heat_df.rename_axis('Truth', axis='rows', inplace=True)
    heat_df.rename_axis('Predicted', axis='columns', inplace=True)
    heat_df.to_csv(f'{out_dir}heat_df.csv')
    annot_mat.to_csv(f'{out_dir}heat_df_annot.csv')
    for cmap in ('Blues', 'Oranges', 'Greens'):
        fig, ax = plt.subplots(figsize=(5,5))
        sns.heatmap(data=heat_df, annot=annot_mat, fmt='s', cmap=cmap)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(title='RDS')
        plt.savefig(f'{out_dir}heatmap_rankdiff_{cmap}.svg', dpi=400)
        plt.close(fig)
