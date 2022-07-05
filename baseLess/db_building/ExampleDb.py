import ZODB, ZODB.FileStorage, BTrees.IOBTree
from os.path import isfile
import numpy as np
import random
from copy import deepcopy

COMPLEMENT_DICT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

class ExampleDb(object):
    """
    A class for a database storing training examples for a neural network
    """
    def __init__(self, **kwargs):

        self._db = None
        self.neg_kmers = dict()  # Dict of lists, indices per encountered negative example k-mer
        self._db_empty = True
        self.nb_pos = 0
        self.nb_neg = 0
        if not isfile(kwargs['db_name']):
            self.target = kwargs['target']
            self.width = kwargs['width']
        self.db_name = kwargs['db_name']

        self.read_only = kwargs.get('read_only', False)
        self.db = self.db_name

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, kmer):
        # If the kmer is already provided as tuple of kmer and complement,
        # Set correct target immediately (kinda ugly but it works)
        if len(kmer) == 2:
            self._target = kmer
        # Otherwise determine complement of the kmer
        else:
            complement = ''.join([COMPLEMENT_DICT[n] for n in kmer][::-1])
            self._target = (kmer, complement)

    def add_training_read(self, training_read, uncenter_kmer):
        """Add training read with positive and negative read examples of the
        target k-mer

        :param training_read: Object containing a training read
        :type training_read: TrainingRead
        :param uncenter_kmer: If set to true, will extract reads where the k-mer
                              is randomly places somewhere in the read. Not
                              always in the center
        :type uncenter_kmer: bool
        """
        with self._db.transaction() as conn:
            # --- add positive examples (if any) ---
            pos_examples = training_read.get_pos(self.target, self.width, uncenter_kmer)
            for i, ex in enumerate(pos_examples):
                conn.root.pos[len(conn.root.pos)] = ex
            nb_new_positives = len(pos_examples)
            # --- update record nb positive examples ---
            if self._db_empty:
                if nb_new_positives > 0:
                    self._db_empty = False
            if not self._db_empty:
                self.nb_pos = conn.root.pos.maxKey()
            # --- add negative examples ---
            neg_examples, neg_kmers = training_read.get_neg(self.target, self.width, len(pos_examples) * 10)  # arbitrarily adding 10x as many neg examples
            for i, ex in enumerate(neg_examples):
                if neg_kmers[i] in self.neg_kmers:
                    self.neg_kmers[neg_kmers[i]].append(self.nb_neg + i)
                else:
                    self.neg_kmers[neg_kmers[i]] = [self.nb_neg + i]
                conn.root.neg[len(conn.root.neg)] = ex
            self.nb_neg += len(neg_examples)
            conn.root.neg_kmers = self.neg_kmers
            return nb_new_positives

    def get_training_set(self, size=None, includes=None):
        """
        Return a balanced subset of reads from the DB
        :param size: number of reads to return
        :param includes: k-mers that should forcefully be included, if available in db
        :return: lists of numpy arrays for training data(x_out) and labels (y_out)
        """
        if includes is None:
            includes = []
        if size is None or size > self.nb_pos or size > self.nb_neg:
            size = min(self.nb_pos, self.nb_neg)
        nb_pos = size // 2
        nb_neg = size - nb_pos
        ps = random.sample(range(self.nb_pos), nb_pos)

        if len(includes):
            forced_includes = []
            forced_includes_list = []
            for k in includes:
                if k in self.neg_kmers:
                    forced_includes_list.append(deepcopy(self.neg_kmers[k]))

            # limit to 20% of neg examples
            nb_neg_forced = nb_neg // 5
            nf_idx = 0
            while len(forced_includes) < nb_neg_forced and len(forced_includes_list):
                forced_includes.append(forced_includes_list[nf_idx].pop())
                if not len(forced_includes_list[nf_idx]):
                    forced_includes_list.remove([])
                nf_idx += 1
                if nf_idx == len(forced_includes_list):
                    nf_idx = 0
            ns = random.sample(range(self.nb_neg), nb_neg - len(forced_includes))
            ns.extend(forced_includes)
        else:
            ns = random.sample(range(self.nb_neg), nb_neg)

        with self._db.transaction() as conn:
            examples_pos = [(conn.root.pos[n], 1) for n in ps]
            examples_neg = [(conn.root.neg[n], 0) for n in ns]
        data_out = examples_pos + examples_neg
        random.shuffle(data_out)
        x_out, y_out = zip(*data_out)
        return x_out, np.array(y_out)

    def pack_db(self):
        self._db.pack()
        with self._db.transaction() as conn:
            self.nb_pos = conn.root.pos.maxKey()
            self.nb_neg = conn.root.neg.maxKey()

    @property
    def db(self):
        return self._db

    @db.setter
    def db(self, db_name):
        """
        Construct ZODB database if not existing, store DB object
        :param db_name: name of new db, including path
        """
        is_existing_db = isfile(db_name)
        storage = ZODB.FileStorage.FileStorage(db_name, read_only=self.read_only)
        self._db = ZODB.DB(storage)
        if is_existing_db:
            with self._db.transaction() as conn:
                self.width = conn.root.width
                self.target = conn.root.target
                self.nb_pos = len(conn.root.pos)
                self.nb_neg = len(conn.root.neg)
                self.neg_kmers = conn.root.neg_kmers
        else:
            with self._db.transaction() as conn:
                conn.root.target = self.target[0]
                conn.root.width = self.width
                conn.root.pos = BTrees.IOBTree.BTree()
                conn.root.neg = BTrees.IOBTree.BTree()
        if self.nb_pos > 0:
            self._db_empty = False
