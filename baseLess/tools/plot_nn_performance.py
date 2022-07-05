import argparse, sys, os
import re
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(f'{__location__}/..')
from low_requirement_helper_functions import parse_input_path

def main():
    parser = argparse.ArgumentParser(description='Plot performance measures for a directory of baseLess NNs')
    parser.add_argument('--nn-dir', type=str, required=True,
                        help='nns directory produced by run_production_pipeline')
    parser.add_argument('--svg', type=str, required=True)
    args = parser.parse_args()

    fn_list = parse_input_path(args.nn_dir, pattern='*performance.pkl')

    performance_dict = {}
    for fn in fn_list:
        with open(fn, 'rb') as fh: cur_dict = pickle.load(fh)
        km = re.search('[^/]+(?=/performance.pkl)', fn).group(0)
        performance_dict[km] = {'accuracy': cur_dict['val_binary_accuracy'][-1],
                                'precision': cur_dict['val_precision'][-1],
                                'recall': cur_dict['val_recall'][-1]}
    performance_df = pd.DataFrame.from_dict(performance_dict).T
    performance_df.to_csv(os.path.splitext(args.svg)[0] + '.csv')

    fig, (ax_pr, ax_acc) = plt.subplots(1,2, figsize=(8.25, 2.9375))

    # precision recall
    sns.scatterplot(x='recall', y='precision', data=performance_df, ax=ax_pr)
    for km, tup in performance_df.iterrows():
        ax_pr.text(x=tup.recall, y=tup.precision, s=km, fontsize=5)

    # accuracy
    sns.violinplot(y='accuracy', data=performance_df, color="0.8", ax=ax_acc)
    sns.stripplot(y='accuracy', data=performance_df, ax=ax_acc)
    for km, tup in performance_df.iterrows():
        ax_acc.text(x=0.1, y=tup.accuracy, s=km, fontsize=5)
    ll = min(performance_df.precision.min(), performance_df.recall.min()) - 0.01
    ax_pr.set_ylim(ll,1); ax_pr.set_xlim(ll,1)
    ax_pr.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(args.svg, dpi=400)
    plt.close()


if __name__ == '__main__':
    main()
