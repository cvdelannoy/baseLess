from scipy.stats import poisson
import matplotlib.pyplot as plt
import numpy as np

"""
Script to estimate the false positive rate of baseless for various values of 
read length, target length, library size and kmer size
"""


class RoughEstimateKmer:
    def __init__(self, target_length: int, k: int, lib_size: int,
                 read_length: int = 0, max_kmers_in_target = np.inf):
        """Create K-mer estimation class

        :param read_length: length of the complete read that goes through a
                            nanopore
        :param target_length: length of the target sequence (e.g. the barcode
                              region)
        :param k: Length of k-mer (typically between 1-15)
        :param lib_size: Number of k-mers for which we will train
                         the neural networks
        :param max_kmers_in_target: Maximum number of k-mers that can be
                                    found in target by neural networks.
                                    I.e. maximum number of models that can
                                    run in parallel
        """
        self.read_length = read_length
        self.target_length = target_length
        self.k = k
        self.lib_size = lib_size
        self.max_kmers_in_target = max_kmers_in_target

        # False positive rate =
        # P(kmer in found by in read)**E(number of k-mers in target)
        self.fp_rate = self.p_kmer_in_read ** self.kmer_count_in_target

    def p_kmer_by_chance(self, length):
        """Probability that a k-mer of length k is found at least once in a
        read of given length.
        """
        # Mu is expected value to be used in Poisson distribution
        mu = length / (4 ** self.k)
        distr = poisson(mu)
        return 1 - distr.cdf(0)

    @property
    def kmer_count_in_target(self):
        """Estimate total number of distinct k-mers from the library that we
        expect to find in target sequence
        """
        # Fraction found = p(k-mer being found in target) * library size
        return min(self.max_kmers_in_target,
                   self.p_kmer_by_chance(self.target_length) * self.lib_size)

    @property
    def p_kmer_in_read(self):
        """Probability that a k-mer is found in read at least once"""
        return self.p_kmer_by_chance(self.read_length)


def plot_fdr(lib_sizes: list, read_lengths: list, target_lengths: list,
             max_kmers_in_target = np.inf):
    """Plot theoretical false discovery rate when recognizing k-mers

    :param lib_sizes: list of library sizes (i.e. number of kmers for which
                      we train models)
    :param read_lengths: List of read lengths of the nanopore
    :param target_lengths: List of possible target lengths
    :param max_kmers_in_target: Maximum number of k-mers that can be
                            found in target by neural networks.
                            I.e. maximum number of models that can
                            run in parallel
    """
    for lib_size in lib_sizes:
        fig, ax = plt.subplots()
        for read_length in read_lengths:
            for target_length in target_lengths:
                x = []
                y1 = []
                for k in range(1, 15):
                    estimator = RoughEstimateKmer(target_length, k, lib_size,
                                                  read_length, max_kmers_in_target)
                    x.append(k)
                    y1.append(estimator.fp_rate)

                ax.plot(x, y1, label=f"{read_length}, {target_length}")
                print(lib_size, read_length, target_length)
                print(f"Lowest fdr: {min(y1)}")
                print(f"At k = {y1.index(min(y1)) + 1}")

        ax.set_title(f"Theoretical FPR at lib size {lib_size}")
        ax.set_ylabel('Probability of false-positive')
        ax.set_xlabel('k')
        plt.legend(title='Read length, target length')
        plt.show()


def kmers_in_target(lib_sizes: list, target_lengths: list,
                    max_kmers_in_target=np.inf):
    """Display how many k-mers of the library will be found in target sequence

    :param lib_sizes: list of library sizes (i.e. number of kmers for which
                      we train models)
    :param target_lengths: List of possible target lengths
    :param max_kmers_in_target: Maximum number of k-mers that can be
                            found in target by neural networks.
                            I.e. maximum number of models that can
                            run in parallel
    """
    fig, ax = plt.subplots()
    for lib_size in lib_sizes:
        for target_length in target_lengths:
            x = []
            y = []
            for k in range(1, 15):
                x.append(k)
                estimator = RoughEstimateKmer(target_length, k, lib_size,
                                              max_kmers_in_target=max_kmers_in_target)
                if k == 8:
                    print(f'{lib_size=},\t{target_length=},\t'
                          f'kmer count: {estimator.kmer_count_in_target}')
                y.append(estimator.kmer_count_in_target)
            ax.plot(x, y, label=f"{lib_size}, {target_length}")
    plt.legend(title='Library size, target size')
    ax.set_title('Expected number of kmers found in target sequence')
    ax.set_xlabel('k')
    ax.set_ylabel('Distinct kmers in target')
    plt.show()


def main():
    library_sizes = [1000, 1500, 2000, 3000]
    read_lengths = [30000, 50000]
    # # Target length: Barcode from BOLD, 16sRRNA, mtDNA, Coronavirus genome
    # target_lengths = [600, 1500, 16000, 30000]
    # Target length: Barcode from BOLD, 16sRRNA, Coronavirus genome
    target_lengths = [600, 1500, 30000]
    # Optional: insert maximum number of k-mers from sequence that we can scan for
    # I.e. how many models can run in parallel
    max_kmers = 64
    kmers_in_target(library_sizes, target_lengths, max_kmers)
    print()
    plot_fdr(library_sizes, read_lengths, target_lengths, max_kmers)


if __name__ == '__main__':
    main()

