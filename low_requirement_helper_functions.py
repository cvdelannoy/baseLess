import os, shutil, re
from glob import glob

import numpy as np

def parse_input_path(in_dir, pattern=None, regex=None):
    if type(in_dir) != list: in_dir = [in_dir]
    out_list = []
    for ind in in_dir:
        if not os.path.exists(ind):
            raise ValueError(f'{ind} does not exist')
        if os.path.isdir(ind):
            ind = os.path.abspath(ind)
            if pattern is not None: out_list.extend(glob(f'{ind}/**/{pattern}', recursive=True))
            else: out_list.extend(glob(f'{ind}/**/*', recursive=True))
        else:
            if pattern is None: out_list.append(ind)
            elif pattern.strip('*') in ind: out_list.append(ind)
    if regex is not None:
        out_list = [fn for fn in out_list if re.search(regex, fn)]
    return out_list


def parse_output_path(location, clean=False):
    """
    Take given path name. Add '/' if path. Check if exists, if not, make dir and subdirs.
    """
    if location[-1] != '/':
        location += '/'
    if clean:
        shutil.rmtree(location, ignore_errors=True)
    if not os.path.isdir(location):
        os.makedirs(location)
    return location


def normalize_raw_signal(raw, norm_method):
    """
    Normalize the raw DAC values

    """
    # Median normalization, as done by nanoraw (see nanoraw_helper.py)
    if norm_method == 'median':
        shift = np.median(raw)
        scale = np.median(np.abs(raw - shift))
    else:
        raise ValueError('norm_method not recognized')
    return (raw - shift) / scale
