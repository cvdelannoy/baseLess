import sys
from os.path import dirname
import numpy as np
import random
from itertools import chain, repeat
from random import choice
# from persistent import Persistent
from pathlib import Path
from low_requirement_helper_functions import normalize_raw_signal
sys.path.append(f'{dirname(Path(__file__).resolve())}/..')


class Read:
    """Class that contains pure reads without any analysis

    :param hdf: HDF file that contains read data
    :type hdf: h5py.File
    :param normalization: how to normalize reads
    :type normalization: str
    """
    def __init__(self, hdf, normalization):
        self.hdf = hdf
        self.normalization = normalization
        self.raw = None

    @property
    def raw(self):
        return self._raw

    @raw.setter
    def raw(self, _):
        raw_varname = self.hdf['Raw/Reads/'].visit(str)
        raw = self.hdf[f'Raw/Reads/{raw_varname}/Signal'][()]
        self._raw = normalize_raw_signal(raw, self.normalization)

    def get_split_raw_read(self, length, stride=5):
        """Split into parts of certain length, last split may be shorter"""
        nrows = ((self.raw.size - length) // stride) + 1
        n = self.raw.strides[0]
        raw_strided = np.expand_dims(np.lib.stride_tricks.as_strided(self.raw, shape=(nrows, length), strides=(stride * n, n)), -1)
        return raw_strided

        # # Uncomment to split up into separate batches
        # split_ids = range(batch_size, math.floor(len(raw_strided)), batch_size)
        # raw_strided_split = np.array_split(raw_strided, split_ids, axis=0)
        # bsl = raw_strided_split[-1].shape[0]
        # if bsl < batch_size:
        #     last_batch = np.zeros((batch_size, length, 1))
        #     last_batch[:bsl, :, :] = raw_strided_split[-1]
        #     raw_strided_split[-1] = last_batch
        # return raw_strided_split, bsl


class TrainingRead(object):
    """Class that contains read with added analysis data, used for training

    :param hdf_path: internal path to analysis files in hdf file
    :type hdf_path: str
    :param kmer_size: size of k-mer
    :type kmer_size: int
    """
    def __init__(self, hdf, normalization, hdf_path, kmer_size):
        """Initialize a training read
        """
        self.hdf = hdf
        self.hdf_path = hdf_path
        self.normalization = normalization

        self.condensed_events = None
        self._event_length_list = None

        self.kmer_size = kmer_size

        self.raw = None
        self.events = None

    def expand_sequence(self, sequence, length_list=None):
        """
        Expand a 1-event-per-item list to a one-raw-data-point-per-item list, using the event lengths derived from
        the basecaller. Uses event length list stored in object if none provided
        """
        if length_list is None:
            return list(chain.from_iterable(repeat(item, duration) for item, duration in zip(sequence,
                                                                                             self._event_length_list)))
        return list(chain.from_iterable(repeat(item, duration) for item, duration in zip(sequence, length_list)))

    @property
    def clipped_bases_start(self):
        # Catches a version change!
        if 'clipped_bases_start' in self.hdf[f'{self.hdf_path}BaseCalled_template/Alignment'].attrs:
            return self.hdf[f'{self.hdf_path}BaseCalled_template/Alignment'].attrs[
                'clipped_bases_start']
        elif 'trimmed_obs_start' in self.hdf[f'{self.hdf_path}BaseCalled_template/Alignment'].attrs:
            return self.hdf[f'{self.hdf_path}BaseCalled_template/Alignment'].attrs[
                'trimmed_obs_start']

    @property
    def clipped_bases_end(self):
        # Catches a version change!
        if 'clipped_bases_end' in self.hdf[f'{self.hdf_path}BaseCalled_template/Alignment'].attrs:
            return self.hdf[f'{self.hdf_path}BaseCalled_template/Alignment'].attrs[
                'clipped_bases_end']
        elif 'trimmed_obs_end' in self.hdf[f'{self.hdf_path}BaseCalled_template/Alignment'].attrs:
            return self.hdf[f'{self.hdf_path}BaseCalled_template/Alignment'].attrs[
                'trimmed_obs_end']

    @property
    def hdf_path(self):
        return self._hdf_path

    @property
    def events(self):
        events, _, _ = zip(*self.condensed_events)
        return self.expand_sequence(events)

    @property
    def start_idx(self):
        _, start_idx, _ = zip(*self.condensed_events)
        return self.expand_sequence(start_idx)

    @property
    def event_length_list(self):
        return self._event_length_list

    @property
    def raw(self):
        return self._raw

    @hdf_path.setter
    def hdf_path(self, pth):
        if pth[-1] != '/':
            pth += '/'
        if pth not in self.hdf:
            raise ValueError('hdf path not in hdf file!')
        if pth + 'BaseCalled_template' not in self.hdf:
            raise ValueError('hdf path in hdf file, but does not contain BaseCalled_template results!')
        self._hdf_path = pth

    @events.setter
    def events(self, _):
        """
        Retrieve k-mers and assign to corresponding raw data points.
        """
        hdf_events_path = f'{self.hdf_path}BaseCalled_template/Events'

        event_states_sl = self.hdf[hdf_events_path]["base"]
        event_states_sl = event_states_sl.astype(str)
        event_list = [''.join(event_states_sl[i:i + self.kmer_size])
                      for i in range(0, event_states_sl.size - self.kmer_size + 1)]
        start_idx_list = self.hdf[hdf_events_path]["start"][:- self.kmer_size + 1]
        # Todo for now we throw away the last couple of k-mers
        event_raw_list = [self.raw[b:e] for b, e in zip(start_idx_list[:-1], start_idx_list[self.kmer_size:])]
        event_length_list = list(self.hdf[hdf_events_path]["length"])

        self.condensed_events = list(zip(event_list,  # k-mers
                                         start_idx_list,  # index of first base in raw read
                                         event_raw_list))  # raw data points in event
        self._event_length_list = event_length_list

    @raw.setter
    def raw(self, _):
        raw_varname = self.hdf['Raw/Reads/'].visit(str)
        raw = self.hdf[f'Raw/Reads/{raw_varname}/Signal'][()]
        first_sample = self.hdf[f'{self.hdf_path}BaseCalled_template/Events'].attrs['read_start_rel_to_raw']
        raw = raw[first_sample-1:]  # NOTE: -1, or you throw away the first sample
        self._raw = normalize_raw_signal(raw, self.normalization)

    def get_pos(self, kmers, width, uncenter_kmer=True, nb=None):
        """Return raw reads of length width with the target kmer in them

        :param kmers: tuple of Kmers to target
        :param width: Width of raw read data to cut out
        :param uncenter_kmer: If set to True; the kmer won't always be centered
                            in the returned read
        :type uncenter_kmer: bool
        :param nb: number of examples to return
        :return: list of raw hits out that contain the k-mer in the read
        """
        # Condensed hits is list of raw reads that contain the kmer.
        # The 1st item the condensed event, the 2nd item is the index of the
        # kmer in the basecalled sequence
        condensed_hits = [(ce, idx) for idx, ce
                          in enumerate(self.condensed_events) if ce[0] in kmers]

        raw_hits_out = []
        for ch, _ in condensed_hits:
            # Data augmentation, works if uncenter_kmer == true:
            # place every example in 5 different positions
            for _ in range(5):
                width_l = (width + 1) // 2  # if width is even, pick point RIGHT of middle
                width_r = width - width_l
                if uncenter_kmer:
                    max_offset = width_r - len(ch[2]) // 2 - 1
                    random_offset = random.randint(-max_offset, max_offset)

                    width_l -= random_offset
                    width_r += random_offset
                mid_idx = ch[1] + len(ch[2])//2
                # Handle edge cases
                if width_l > mid_idx:
                    # Cannot cut off enough on left side
                    width_l = mid_idx
                    width_r = width - width_l
                elif width_r > len(self.raw) - mid_idx:
                    # Cannot cut off enough on right side
                    width_r = len(self.raw) - mid_idx
                    width_l = width - width_r
                candidate_raw = self.raw[mid_idx - width_l:
                                         mid_idx + width_r]
                kmer_in_read = np.all(np.in1d(ch[2], candidate_raw))
                assert kmer_in_read, 'K-mer not in positive read'
                raw_hits_out.append(candidate_raw)
        return raw_hits_out

    def get_neg(self, kmers, width, nb):
        idx_list = list(range(len(self.condensed_events)))
        # Indices of target kmer in raw read
        pos_events = [ce for ce in self.condensed_events
                      if ce[0] in kmers]
        width_l = (width + 1) // 2  # if width is even, pick point RIGHT of middle
        width_r = width - width_l
        raw_hits_out = []
        raw_kmers_out = []
        while len(raw_hits_out) < nb and len(idx_list) > 0:
            cur_idx = choice(idx_list)
            cur_condensed_event = self.condensed_events[cur_idx]
            # Explicitly convert to python integers to prevent overflow error
            distances_to_kmer = [abs(int(cur_condensed_event[1]) - int(pos_event[1]))
                                 for pos_event in pos_events]
            pos_kmer_seqs = [pos_event[2] for pos_event in pos_events]
            # Make sure the negative examples are far enough away from
            # the target k mer (manually add 30 to be extra safe).
            idx_list.remove(cur_idx)
            if np.all(np.array(distances_to_kmer) > width):
                mid_idx = cur_condensed_event[1] + len(cur_condensed_event[2])//2
                candidate_raw = self.raw[mid_idx - width_l:mid_idx + width_r]
                target_kmer_in_read = np.all([np.in1d(pos_raw, candidate_raw)
                                              for pos_raw in pos_kmer_seqs])
                if target_kmer_in_read:
                    # Something went wrong (probably edge case (that I should fix)), try again
                    continue
                raw_hits_out.append(candidate_raw)
                raw_kmers_out.append(cur_condensed_event[0])

        return raw_hits_out, raw_kmers_out
