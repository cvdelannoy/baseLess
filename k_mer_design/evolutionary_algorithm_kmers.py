"""Script that creates library of maximally dissimilar k-mers.
Needs scrappy to be installed. And can be run from the command line

Based largely on the following paper:

Doroschak, K., Zhang, K., Queen, M. et al.
Rapid and robust assembly and decoding of molecular tags with DNA-based
nanopore signatures. Nat Commun 11, 5454 (2020).
https://doi.org/10.1038/s41467-020-19151-8
"""

import argparse
import os
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
from _ucrdtw import ucrdtw
import json


sns.set(font_scale=1.5, style="white")


def s_w(seq1, seq2, cost_fn={"match": 1, "mismatch": -1, "gap": -4}):
    '''Smith-Waterman implementation (also use this for TAing 427)'''
    nrows = len(seq1) + 1  # + 1 is to accommodate the 0th row
    ncols = len(seq2) + 1
    dp = np.zeros((nrows, ncols))

    def score():
        best_score = -np.Inf
        best_pos = (0, 0)
        for row_i in range(1, nrows):
            for col_j in range(1, ncols):
                score_ij = _score_ij(row_i, col_j)
                dp[row_i, col_j] = score_ij
                if score_ij >= best_score:
                    best_score = score_ij
                    best_pos = (row_i, col_j)
        return best_pos, best_score

    def _score_ij(i, j):
        if seq1[i - 1] == seq2[j - 1]:
            match_cost = cost_fn["match"]
        else:
            match_cost = cost_fn["mismatch"]

        up = dp[i - 1, j] + cost_fn["gap"]  # anything but diag must be a gap
        left = dp[i, j - 1] + cost_fn["gap"]
        dia = dp[i - 1, j - 1] + match_cost

        return max(up, left, dia)

    def traceback(best_pos):
        '''Ties prefer diagonal > up > left in this implementation'''
        seq1_out, seq2_out, match_str = "", "", ""
        row_i, col_j = best_pos

        # Deal with uneven sequences
        if row_i == col_j and row_i < nrows - 1 and col_j < ncols - 1:
            seq1_out = seq1[row_i:]
            seq2_out = seq2[col_j:]
            for i in range(row_i, nrows - 1):
                match_str += ":"
        if row_i != col_j and row_i < nrows - 1:
            seq1_out = seq1[row_i:]
            for i in range(row_i, nrows - 1):
                match_str += " "
                seq2_out += "-"
        if row_i != col_j and col_j < ncols - 1:
            seq2_out = seq2[col_j:]
            for i in range(col_j, ncols - 1):
                match_str += " "
                seq1_out += "-"

        # Traceback
        last_dia_s1 = 0
        last_dia_s2 = 0
        while row_i and col_j:  # end when either is 0
            up = dp[row_i - 1, col_j]
            left = dp[row_i, col_j - 1]
            dia = dp[row_i - 1, col_j - 1]

            # Case 1: diagonal
            if dia >= up and dia >= left:
                row_i -= 1
                col_j -= 1
                last_dia_s1 = row_i
                last_dia_s2 = col_j
                seq1_out = seq1[row_i] + seq1_out
                seq2_out = seq2[col_j] + seq2_out
                if seq1[row_i] == seq2[col_j]:
                    match_str = "|" + match_str
                else:
                    match_str = ":" + match_str
            # Case 2: up
            elif up >= left:
                row_i -= 1
                seq1_out = seq1[row_i] + seq1_out
                seq2_out = "-" + seq2_out
                match_str = " " + match_str
            # Case 3: left
            else:
                col_j -= 1
                seq1_out = "-" + seq1_out
                seq2_out = seq2[col_j] + seq2_out
                match_str = " " + match_str

        # Deal with uneven sequences
        if 0 < row_i:
            seq1_out = seq1[:row_i] + seq1_out
            for i in range(0, row_i):
                seq2_out = "-" + seq2_out
                match_str = " " + match_str
        if 0 < col_j:
            seq2_out = seq2[:col_j] + seq2_out
            for i in range(0, col_j):
                seq1_out = "-" + seq1_out
                match_str = " " + match_str

        return seq1_out, seq2_out, match_str, last_dia_s1, last_dia_s2

    best_pos, best_score = score()
    seq1_out, seq2_out, match_str, last_dia_s1, last_dia_s2 = traceback(
        best_pos)
    return best_pos, best_score, "\n".join([seq1_out, match_str, seq2_out]), \
           last_dia_s1, last_dia_s2


def revcomp(seq):
    seq = seq.upper()
    seq = seq.replace("A", "X")
    seq = seq.replace("T", "A")
    seq = seq.replace("X", "T")
    seq = seq.replace("C", "X")
    seq = seq.replace("G", "C")
    seq = seq.replace("X", "G")
    return seq[::-1]


def find_gc_content(sequence):
    gc_content = 1. * (sequence.count("G") +
                       sequence.count("C")) / len(sequence)
    return gc_content


def check_valid_seq(sequence, remove_if_outer10_percent=False):
    """We will omit the GC range limit in this simulation, originally the gc percentage had to be between 0.3 and 0.7
    """

    valid = True
    reason = []

    # Check homopolymers
    for h in homopolymers:
        if h in sequence:
            reason.append("Contains a homopolymer")
            valid = False
            break

    # If remove_if_outer10_percent is set to true; remove sequences that
    # have an abundance in the outer 10% of all 16s sequences
    if remove_if_outer10_percent and sequence in OUTER10_PERCENT:
        reason.append('Extreme presence or absence')
        valid = False

    return valid, reason


def get_dynamic_range(squiggle):
    dr = np.abs(np.max(squiggle) - np.min(squiggle))
    return dr


def check_valid_squig(squiggle):
    valid_dr = ("X", "Y")
    dr = get_dynamic_range(squiggle)
    if dr < valid_dr[0] or dr > valid_dr[1]:
        return False
    return True


def mutate(sequence, n_mutations=1):
    assert n_mutations <= len(sequence)
    new_sequence = list(sequence)
    nts = ("A", "C", "G", "T")
    ix_to_mutate = np.random.choice(
        range(len(new_sequence)), size=n_mutations, replace=False)
    for i in ix_to_mutate:
        possible_bases = list(nts)
        possible_bases.remove(new_sequence[i])
        new_sequence[i] = np.random.choice(possible_bases, size=1)[0]
    new_sequence = "".join([str(x) for x in new_sequence])

    # Test
    n_diff = 0
    for i in range(len(sequence)):
        if sequence[i] != new_sequence[i]:
            n_diff += 1
    assert n_diff == n_mutations
    return new_sequence


def mutate_adjacent(sequence, n_mutations=1):
    """This was modified by me because it was broken"""
    assert n_mutations <= len(sequence)
    new_sequence = list(sequence)
    nts = ("A", "C", "G", "T")
    ix_to_mutate = np.random.choice(len(new_sequence) - n_mutations + 1)

    for j in range(n_mutations):
        possible_bases = list(nts)
        possible_bases.remove(new_sequence[ix_to_mutate + j])
        new_sequence[int(ix_to_mutate + j)] = \
            np.random.choice(possible_bases, size=1)[0]
    new_sequence = "".join([str(x) for x in new_sequence])

    # Test
    n_diff = 0
    for i in range(len(sequence)):
        if sequence[i] != new_sequence[i]:
            n_diff += 1
    assert n_diff == n_mutations
    return new_sequence


def simulate_squiggle(sequence):
    """
    Given a sequence as string; create a dataframe that describes
    its MinION squiggle
    """
    rand = np.random.randint(0, 500000)
    temp_fasta_fname = sequence + "_" + str(rand) + "_temp.fa"
    temp_scrappie_fname = sequence + "_" + str(rand) + "_temp.scrappie"
    with open(temp_fasta_fname, "w") as f:
        f.write(">temp\n%s\n" % (sequence))
    scrappie_str = "scrappie squiggle -o %s %s" % (
        temp_scrappie_fname, temp_fasta_fname)
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
    df = pd.DataFrame(scrappie_sim,
                      columns=["pos", "base", "current", "sd", "dwell"])
    df = df.astype({"pos": int, "base": str, "current": float, "sd": float,
                    "dwell": float})
    return df


def calc_dtw_from_seq(seq_1, seq_2, warp_width=0.1):
    """
    Given two sequences, calculate their dtw distance
    """
    scrappie_df_1 = simulate_squiggle(seq_1)
    scrappie_df_2 = simulate_squiggle(seq_2)
    squig1 = list(scrappie_df_1["current"])
    squig2 = list(scrappie_df_2["current"])
    dist = calc_dtw(squig1, squig2, warp_width=warp_width)
    return dist


def calc_dtw(scrappie_df_1, scrappie_df_2, warp_width=0.1):
    """Given two lists of currents (the 'current column' of the scrappie df),
     calculate their dtw distance"""
    _, dtw_dist = ucrdtw(scrappie_df_1, scrappie_df_2, warp_width, False)
    dtw_dist = np.float32(dtw_dist)
    return dtw_dist


def load_squiggles_from_csv(file_path):
    """Get CSV of all kmers and their squiggles and convert it to dict"""
    df = pd.read_csv(file_path, index_col=0, header=None)
    df = df.transpose()
    return df.to_dict('list')


def initialize_distance(sequences):
    """Given sequences and dictionary with all squiggles, create a distance
    matrix that shows DTW distance"""
    n_seqs = len(sequences)
    D = np.zeros((n_seqs, n_seqs))
    squiggles = [None for _ in range(n_seqs)]
    for j, seq1 in enumerate(sequences):
        for i, seq2 in enumerate(sequences):
            if i >= j:
                continue
            else:
                dtw_dist = calc_dtw(SQUIGGLE_DICT[seq1], SQUIGGLE_DICT[seq2])
                print(j / n_seqs, seq1, seq2, dtw_dist)
                D[i, j] = dtw_dist
                D[j, i] = dtw_dist
    return D


def recalc_distance(D, row_ix, new_squiggle, seq_list):
    """Update distance matrix d for new squiggle
    """
    D_update = D.copy()
    for j in range(D.shape[0]):
        if j == row_ix:
            continue
        kmer_to_compare = seq_list[j]
        squiggle_j = SQUIGGLE_DICT[kmer_to_compare]
        dtw_dist = calc_dtw(new_squiggle, squiggle_j)
        D_update[j, row_ix] = dtw_dist
        D_update[row_ix, j] = dtw_dist
    return D_update


def calculate_new_distances(new_seq, sequences, ignore_ix=None,
                            sw_cost_fn={"match": 1, "mismatch": -1,
                                        "gap": -4}):
    """Given a new sequence, calculate smith waterman distances to other sequences,
    but ignores seqs with index in ignore_ix"""
    d_sw = np.zeros(len(sequences))
    for seq_j in range(len(sequences)):
        if ignore_ix is not None and seq_j in ignore_ix:
            continue
        _, sw, _, _, _ = s_w(new_seq, sequences[seq_j], cost_fn=sw_cost_fn)
        d_sw[seq_j] = sw
    return d_sw


def check_improvement(d_old, d_new):
    # d is a single row of D
    identity_ix = np.where(np.logical_and(d_old == 0, d_new == 0))
    d_old = np.delete(d_old, identity_ix)
    d_new = np.delete(d_new, identity_ix)
    min_improved = min(d_old) < min(d_new)
    avg_improved = np.mean(d_old) < np.mean(d_new)
    return min_improved and avg_improved


def make_random_seq(length=40):
    """Generates dna string of nonrepeating bases"""

    def norepeat_DNA():
        seq = ["A", "C", "G", "T"]
        last_bp = np.random.choice(seq)
        yield last_bp
        while True:
            next_bp = np.random.choice(seq)
            if next_bp != last_bp:
                last_bp = next_bp
                yield next_bp

    g = norepeat_DNA()
    seq = ""
    for _ in range(length):
        seq += next(g)
    return seq


def initialise_kmers(kmer_size, lib_size, match_score, mismatch_score,
                     gap_score, max_sw_sim_init, max_tries=200):
    """Create kmer library where kmers are sufficiently unique

    :param kmer_size: Length of a kmer (we used 8)
    :param lib_size: Total number of different kmers to output
    :param match_score: Smith waterman match score
    :param mismatch_score: Smith waterman mismatch score
    :param gap_score: Smith waterman gap score
    :param max_sw_sim_init: Maximum smith waterman similarity at initialisation
    :param max_tries: Maximum attempts per kmer at generating a valid one
    :return: list of kmers in the library
    """

    # Create initial kmers here
    starting_seqs = [make_random_seq(kmer_size) for _ in range(lib_size)]
    new_starting_seqs = starting_seqs[:]
    for seq_i, seq in enumerate(starting_seqs):
        # Check if it is too similar to an other seq
        too_similar = False
        d_sw = calculate_new_distances(seq, new_starting_seqs,
                                       ignore_ix=[seq_i],
                                       sw_cost_fn={"match": match_score,
                                                   "mismatch": mismatch_score,
                                                   "gap": gap_score})
        if np.max(d_sw) > max_sw_sim_init:
            too_similar = True

        valid, reasons = check_valid_seq(seq)
        print(seq_i, valid, too_similar, reasons)
        tries = 0
        new_seq = seq
        while (not valid or too_similar) and tries < max_tries:
            too_similar = True
            tries += 1
            new_seq = make_random_seq(kmer_size)
            if new_seq in new_starting_seqs:
                valid = False
                print('Already present')
                continue
            valid, _ = check_valid_seq(new_seq)

            d_sw = calculate_new_distances(new_seq, new_starting_seqs,
                                           ignore_ix=[seq_i],
                                           sw_cost_fn={"match": match_score,
                                                       "mismatch": mismatch_score,
                                                       "gap": gap_score})
            if np.max(d_sw) <= max_sw_sim_init:
                too_similar = False
        else:
            print(f'Performed {tries} tries')
            new_starting_seqs[seq_i] = new_seq
    starting_seqs = new_starting_seqs[:]
    return starting_seqs


def initialise_sw_distances(starting_seqs, match_score,
                            mismatch_score, gap_score, sw_print_cutoff):
    """

    :param starting_seqs:
    :param match_score:
    :param mismatch_score:
    :param gap_score:
    :param sw_print_cutoff:
    :return:
    """
    lib_size = len(starting_seqs)
    D_sw = np.zeros((lib_size, lib_size))
    D_sw_flat = []
    for j in range(lib_size):
        for i in range(lib_size):
            if i >= j:
                continue
            else:
                _, sw, a, _, _ = s_w(starting_seqs[j], starting_seqs[i],
                                     cost_fn={"match": match_score,
                                              "mismatch": mismatch_score,
                                              "gap": gap_score})
                D_sw[i, j] = sw
                D_sw[j, i] = sw
                D_sw_flat.append(sw)
                if sw >= sw_print_cutoff:  # Print seqs that are similar to inspect manually, value of 7 seems little to high, too
                    print(sw)
                    print(i, j)
                    print()
                    print(a)
                    print()
    return D_sw, D_sw_flat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform evolutionary '
                                                 'algorithm to generate '
                                                 'library of kmers')
    parser.add_argument('--squiggle-dict', required=True,
                        help='Path to CSV containing all simulated squiggles')
    parser.add_argument('--library-size', required=True, type=int,
                        help='Number of different k-mers to generate')
    parser.add_argument('-o', '--out-directory', required=True,
                        help='Output directory')
    parser.add_argument('--remove-outer10-percent', action='store_true',
                        default=False, help='If this flag is provided, remove '
                                            'kmers that have an abundance in '
                                            'the outer 10%% in 16s rRNA '
                                            'sequences')

    args = parser.parse_args()

    if not os.path.exists(args.out_directory):
        os.mkdir(args.out_directory)

    homopolymers = ["AAAAA", "TTTTT", "GGGG", "CCCC"]
    if args.remove_outer10_percent:
        with open('./16sNCBIdatabase/outer10kmers.json', 'r') as f:
            f = f.read()
            OUTER10_PERCENT = json.loads(f)

    SQUIGGLE_DICT = load_squiggles_from_csv(args.squiggle_dict)

    # Set parameters
    lib_size = args.library_size
    kmer_size = 8
    # Maximum smith waterman alignment score at kmer initiliasation step
    max_sw_sim_init = 6
    # Alignment parameters
    match_score = 1
    mismatch_score = -1
    gap_score = -4

    print('Creating starting sequences')
    starting_seqs = initialise_kmers(kmer_size, lib_size, match_score,
                                     mismatch_score, gap_score,
                                     max_sw_sim_init, max_tries=200)
    assert len(starting_seqs) == len(set(starting_seqs)), \
        "Starting sequences contain duplicates"
    print('Creating initial DTW distance matrix')
    D = initialize_distance(starting_seqs)

    # Save to CSV here
    distance_df = pd.DataFrame(D)
    distance_df.columns = starting_seqs
    distance_df.index = starting_seqs
    distance_df.to_csv(os.path.join(args.out_directory,
                                    f'InitialDistancesOf{kmer_size}mers.csv'))

    print('Calculating all smith waterman distances')
    D_sw, D_sw_flat = initialise_sw_distances(starting_seqs, match_score,
                                              mismatch_score, gap_score,
                                              sw_print_cutoff=7)

    # # Uncomment to save images
    # linkage = hc.linkage(sp.distance.squareform(distance_df), method='average')
    # sns_plot = sns.clustermap(distance_df, row_linkage=linkage,
    #                           col_linkage=linkage)
    # sns_plot.savefig(os.path.join(args.out_directory,
    #                               'Cluster_DTW_distances_of_initial_kmers.png'))
    #
    # linkage = hc.linkage(sp.distance.squareform(D_sw+1, checks=False),
    #                      method='complete')
    # plot = sns.clustermap(D_sw+1, row_linkage=linkage, col_linkage=linkage)
    # plot.savefig(os.path.join(args.out_directory,
    #                           'Cluster_SW_distances_of_initial_kmers.png'))

    # Parameters TODO: make these not be hardcoded, but for now they work
    n_rounds = 20  # make  5 or more?
    # From this round on, switch from 2 nucleotides per mutation to only one
    start_finetune_round_ix = 14
    max_sw_sim = 6  # Throw away new seq if it scores above this score
    max_tries = 50  # How many attempts to retry improving per sequence

    seqs_after_rounds = [starting_seqs]
    D_after_rounds = [D]
    D_sw_after_rounds = [D_sw]
    triu = np.triu_indices(n=len(starting_seqs), k=1)
    print("Starting summary:\nmin:  %0.4f\nmean: %0.4f"
          % (np.min(D[triu].flatten()), np.mean(D[triu].flatten())))

    ## Uncomment to save the images
    # model_name = "evolve_from_v3_sw_2adj"  # Squiggle images will be saved here
    # ! rm -rf {model_name}
    # os.makedirs(model_name)

    # for seq_i in range(len(starting_seqs)):
    #     os.makedirs("%s/%d" % (model_name, seq_i))

    for round_i in range(n_rounds):
        # How many nucleotides to change per mutation
        mutation_no = 2 if round_i < start_finetune_round_ix else 1
        print("============== Round %d ==============" % round_i)
        current_seqs = seqs_after_rounds[round_i][:]
        #    current_dfs = scrappie_after_rounds[round_i][:]
        current_D = D_after_rounds[round_i].copy()
        current_D_sw = D_sw_after_rounds[round_i].copy()

        if round_i % 3 == True and round_i != 0:
            D_sw = np.zeros((lib_size, lib_size))
            for j in range(lib_size):
                for i in range(lib_size):
                    if i >= j:
                        continue
                    else:
                        _, sw, a, _, _ = s_w(current_seqs[j], current_seqs[i],
                                             cost_fn={"match": match_score,
                                                      "mismatch": mismatch_score,
                                                      "gap": gap_score})
                        D_sw[i, j] = sw
                        D_sw[j, i] = sw
        current_D_sw = D_sw

        for count, seq_i in enumerate(
                np.random.choice(range(len(starting_seqs)),
                                 size=len(starting_seqs), replace=False)):
            print(
                f'Progress: {((count / len(starting_seqs)) * 100):.2f} percent')
            improved = False
            tries = 0
            past_tries = []
            assert len(current_seqs[seq_i]) == kmer_size

            while not improved and tries < max_tries:
                tries += 1
                new_seq = mutate_adjacent(current_seqs[seq_i],
                                          n_mutations=mutation_no)

                assert len(new_seq) == kmer_size
                # Check correct number of mutations
                assert sum([x != y for x, y in zip(new_seq, current_seqs[
                    seq_i])]) == mutation_no

                if new_seq in past_tries:
                    continue

                past_tries.append(new_seq)
                d_sw_prev = current_D_sw[seq_i, :]

                ignore_ix = [seq_i]
                # print(len(ignore_ix))

                d_sw = calculate_new_distances(new_seq, current_seqs,
                                               ignore_ix=ignore_ix,
                                               sw_cost_fn={
                                                   "match": match_score,
                                                   "mismatch": mismatch_score,
                                                   "gap": gap_score})
                if np.max(d_sw) > max_sw_sim:
                    # print("Too similar!", np.max(d_sw))
                    continue

                valid, reasons = check_valid_seq(new_seq,
                                                 remove_if_outer10_percent=args.remove_outer10_percent)
                if valid:
                    new_squiggle = SQUIGGLE_DICT[new_seq]
                    new_D = recalc_distance(current_D, seq_i, new_squiggle,
                                            current_seqs)
                    improved = check_improvement(current_D[seq_i, :],
                                                 new_D[seq_i, :])

            if improved:
                print(f"Improved {seq_i}")
                current_seqs[seq_i] = new_seq
                # current_dfs[seq_i] = new_df
                current_D = new_D.copy()
                current_D_sw[seq_i, :] = d_sw
                current_D_sw[:, seq_i] = d_sw
            else:
                print(
                    f"No improvement (round {round_i}, seq {seq_i}, tries {tries})")

        #         Uncomment this to save figures
        #         fig, ax = plot_scrappie_squiggle(current_dfs[seq_i])
        #         fig.tight_layout()
        #         fig.savefig("%s/%d/%d.png" % (model_name, seq_i, round_i))
        #         plt.close(fig)

        seqs_after_rounds.append(current_seqs)
        #     scrappie_after_rounds.append(current_dfs)
        D_after_rounds.append(current_D.copy())
        D_sw_after_rounds.append(current_D_sw.copy())
        print("Improvement summary:\nmin:  %0.4f\nmean: %0.4f" % (
            np.min(current_D[triu].flatten()),
            np.mean(current_D[triu].flatten())))

        df = pd.DataFrame(current_D)
        df.columns = current_seqs
        df.index = current_seqs
        df.to_csv(os.path.join(args.out_directory,
                               f'1500libraryOf8mers_{round_i}'
                               f'_rounds_DTW.csv'))

        df2 = pd.DataFrame(current_D_sw)
        df2.columns = current_seqs
        df2.index = current_seqs
        df2.to_csv(os.path.join(args.out_directory,
                                f'1500libraryOf8mers_{round_i}'
                                f'_rounds_SW.csv'))

