"""Script that generates squiggles for k-mers. Needs scrappy to be installed.

Based largely on the following paper:

Doroschak, K., Zhang, K., Queen, M. et al.
Rapid and robust assembly and decoding of molecular tags with DNA-based
nanopore signatures. Nat Commun 11, 5454 (2020).
https://doi.org/10.1038/s41467-020-19151-8
"""


import os
import seaborn as sns
import numpy as np
import pandas as pd
import itertools
import multiprocessing
import argparse

sns.set(font_scale=1.5, style="white")


def simulate_squiggle(sequence: str):
    """Given a sequence, generate a dataframe that describes the squiggle
    Note: only seems to work for len(sequence)>2"""
    print(sequence)
    rand = np.random.randint(0, 500000)
    temp_fasta_fname = "./temp/" + sequence + "_" + str(rand) + "_temp.fa"
    temp_scrappie_fname = "./temp/" + sequence + "_" + str(rand) \
                          + "_temp.scrappie"
    with open(temp_fasta_fname, "w+") as f:
        f.write(f">temp\n{sequence}\n")
    scrappie_str = f"scrappie squiggle -o {temp_scrappie_fname} " \
                   f"{temp_fasta_fname}"
    os.system(scrappie_str)
    os.remove(temp_fasta_fname)

    with open(temp_scrappie_fname, "r") as f:
        scrappie_lines = f.readlines()
    os.remove(temp_scrappie_fname)

    scrappie_sim = []
    seq_name = None
    df = None
    for i, line in enumerate(scrappie_lines):
        line = line.strip()
        if line.startswith("#"):
            seq_name = line
        elif line.startswith("pos"):
            continue
        else:
            scrappie_sim.append(line.split("\t"))
    df = pd.DataFrame(scrappie_sim, columns=["pos", "base", "current", "sd",
                                             "dwell"])
    df = df.astype({"pos": int, "base": str, "current": float, "sd": float,
                    "dwell": float})
    return df


def generate_kmers(k):
    """Generate all possible kmers of length k

    :param k: integer, length of kmer
    :return: list of strings with the kmers
    """
    bases = ['A', 'T', 'C', 'G']
    return [''.join(comb) for comb in itertools.product(bases, repeat=k)]


def parallel_simulate_all_kmer_squiggles(k, parallel_count):
    """For all kmers of length k, simulate their squiggle and save to file

    :param k: length of k-mer
    :param parallel_count: how many threads to run in parallel
    :return: dictionary with kmer as key and current as value
    """
    kmer_list = generate_kmers(k)

    with multiprocessing.Pool(parallel_count) as pool:
        scrappie_dfs = pool.map(simulate_squiggle, kmer_list)
    # Create dict with sequence as key and currents as values
    squiggle_dict = {''.join(df['base'].tolist()): list(df['current']) for df
                     in scrappie_dfs}

    df = pd.DataFrame.from_dict(squiggle_dict, orient='index')
    df.to_csv(f'all{k}mer_squiggles.csv', header=False)

    return squiggle_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate k-mer squiggles')
    parser.add_argument('k', type=int, help='Length of k-mers to simulate')
    parser.add_argument('parallel_count', type=int,
                        help='How many threads to run in parallel')
    args = parser.parse_args()
    if not os.path.exists('./temp'):
        os.mkdir('temp')
    parallel_simulate_all_kmer_squiggles(k=args.k,
                                         parallel_count=args.parallel_count)
    os.rmdir('temp')
