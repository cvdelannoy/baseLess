import numpy as np


def npz_to_tf(npz, max_sequence_length, var_names=['raw']):
    """Turn npz-files into tf-readable objects
    :param npz: npz-file storing raw data and labels

    """
    with np.load(npz) as npz_file:
        x = [npz_file[var_name] for var_name in var_names]
        x = np.stack(x, axis=1)
        y = npz_file['labels']
        if x.size == 1:
            sequence_length = 1
        else:
            sequence_length = x.shape[0]
        if sequence_length > max_sequence_length:
            x = x[:max_sequence_length, :]
            y = y[:max_sequence_length]
            sequence_length = max_sequence_length

    return x, y, sequence_length


def npz_to_tf_and_kmers(npz, max_sequence_length=np.inf, var_names=['raw'], target_kmer=None):
    """Turn npz-files into tf-readable objects
    :param npz: npz-file storing raw data and labels

    """
    with np.load(npz) as npz_file:
        x = [npz_file[var_name] for var_name in var_names]
        x = np.stack(x, axis=1)
        # y = npz_file['labels']
        kmers = npz_file['base_labels']
        if x.size == 1:
            sequence_length = 1
        else:
            sequence_length = x.shape[0]
        if sequence_length > max_sequence_length:
            target_bool = np.in1d(kmers, target_kmer)
            if target_kmer and np.any(target_bool):
                start_idx = np.argwhere(target_bool).min()
                oh = max_sequence_length // 2
                sidx = np.arange(max(0, start_idx - oh), min(start_idx + oh, len(kmers)))
            else:
                sidx = np.arange(0, max_sequence_length)

            x = x[sidx, :]
            # y = y[:max_sequence_length]
            kmers = kmers[sidx]
            sequence_length = max_sequence_length

    return x, sequence_length, kmers


def add_to_npz(npz, new_file, new_arrays, new_array_names):
    if new_arrays is not list:
        new_arrays = list(new_arrays)
    if new_array_names is not list:
        new_array_names = list(new_array_names)
    if len(new_arrays) != len(new_array_names):
        raise ValueError('not same number of arrays and array names given.')

    with np.load(npz) as npz_file:
        npz_new = dict(npz_file)
    for new_array, name in zip(new_arrays, new_array_names):
        npz_new[name] = new_array
    np.savez(new_file, **npz_new)
