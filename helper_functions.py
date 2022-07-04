import numpy as np
import time
import warnings
import re
import h5py

from math import nan, log
from pathlib import Path
from statistics import median

from db_building.ExampleDb import ExampleDb
from contextlib import closing

from bokeh.models import ColumnDataSource, LinearColorMapper, LabelSet, Range1d
from bokeh.plotting import figure

from math import pi
from datetime import datetime

from low_requirement_helper_functions import parse_input_path

wur_colors = ['#E5F1E4', '#3F9C35']
categorical_colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072']
continuous_colors = ['#ffffff', '#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84',
                     '#fc8d59', '#ef6548', '#d7301f', '#990000']


def get_full_dbs(in_dir):
    """Given directory of kmer dbs, return which databases contain positive reads

    :param in_dir: directory with subdirectories that belong to kmer dbs
    :type in_dir: str
    :return: Now prints path to directory if it finds reads in database
    """
    print('db_path, number of positive examples')
    db_paths = Path(in_dir).iterdir()
    for db_path in db_paths:
        try:
            db, _ = load_db(str(db_path), read_only=True)
            if db.nb_pos > 0:
                print(f"{db_path}, {db.nb_pos}")
        except KeyError:
            continue


def load_db(db_dir, read_only=False):
    """Load database from given directory

    :param db_dir: path to directory, must contain a 'db.fs' file
    :type db_dir: str
    :param read_only: If database should be read only or not
    :type read_only: bool
    :return: database and squiggles
    """
    if db_dir[-1] != '/':
        db_dir += '/'
    db = ExampleDb(db_name=db_dir + 'db.fs', read_only=read_only)
    squiggles = parse_input_path(db_dir + 'test_squiggles')
    return db, squiggles


def safe_cursor(conn, comm, read_only=True, retries=1000):
    if read_only:
        with conn, closing(conn.cursor()) as c:
            c.execute(comm)
            out_list = c.fetchall()
        return out_list
    else:
        with conn, closing(conn.cursor()) as c:
            for _ in range(retries):
                try:
                    c.execute(comm)
                    return
                except:
                    time.sleep(0.01)
            raise TimeoutError('writing to sql table failed')


def plot_timeseries(raw, base_labels, posterior, y_hat, start=0, nb_classes=2, reflines=(0.90, 0.95, 0.99)):

    nb_points = y_hat.size
    scaling = np.abs(raw.max() - raw.min()) / np.abs(posterior.max() - posterior.min())
    posterior = posterior * scaling
    translation = - posterior.min() + raw.min()
    posterior = posterior + translation

    reflines = np.array([log(rl / (1-rl)) for rl in reflines])
    reflines = reflines * scaling + translation
    source_dict = dict(
        raw=raw[:nb_points],
        posterior=posterior,
        event=list(range(len(y_hat))),
        cat=y_hat,
        cat_height=np.repeat(np.mean(raw), len(y_hat))
    )
    for ri, r in enumerate(reflines):
        source_dict[f'r{ri}'] = np.repeat(r, len(posterior))

    # Main data source
    source = ColumnDataSource(source_dict)

    # Base labels stuff
    base_labels_condensed = [base_labels[0]]
    bl_xcoords = []
    event_ends = []
    new_range = []
    for idx, bl in enumerate(base_labels):
        if bl == base_labels_condensed[-1]:
            new_range.append(idx)
        else:
            base_labels_condensed.append(bl)
            bl_xcoords.append(median(new_range))
            event_ends.append(idx)
            new_range = [idx]
    bl_xcoords.append(median(new_range))
    event_ends.append(new_range[-1])

    bl_source = ColumnDataSource(dict(
        base_labels=base_labels_condensed,
        event_ends= event_ends,
        x=bl_xcoords,
        y= np.repeat(raw.max(), len(bl_xcoords))
    ))

    base_labels_labelset = LabelSet(x='x', y='y', text='base_labels',
                                    source=bl_source,
                                    text_baseline='middle',
                                    angle=0.25*pi)

    # Plotting
    ts_plot = figure(title='Classified time series')
    ts_plot.grid.grid_line_alpha = 0.3
    ts_plot.xaxis.axis_label = 'nb events'
    ts_plot.yaxis.axis_label = 'current signal'
    y_range = raw.max() - raw.min()

    colors=wur_colors
    col_mapper = LinearColorMapper(palette=colors, low=0, high=nb_classes-1)
    ts_plot.rect(x='event', y='cat_height', width=1.05, height=y_range, source=source,
                 fill_color={
                     'field': 'cat',
                     'transform': col_mapper
                 },
                 line_color=None)

    # Event ending lines
    ts_plot.rect(x='event_ends',
                 y=raw.mean(),
                 height=y_range, width=0.01, line_color='black', source=bl_source)


    ts_plot.add_layout(base_labels_labelset)
    ts_plot.line(x='event', y='raw', source=source)
    ts_plot.line(x='event', y='posterior', color='red', source=source)
    for i in range(len(reflines)):
        ts_plot.line(x='event', y=f'r{i}', color='grey', source=source)
    ts_plot.plot_width = 1500
    ts_plot.plot_height = 500
    ts_plot.x_range = Range1d(start, start+100)
    return ts_plot


def plot_roc_curve(roc_list):
    tpr, tnr, epoch = zip(*roc_list)
    roc_plot = figure(title='ROC')
    roc_plot.grid.grid_line_alpha = 0.3
    roc_plot.xaxis.axis_label = 'FPR'
    roc_plot.yaxis.axis_label = 'TPR'

    col_mapper = LinearColorMapper(palette=categorical_colors, low=1, high=max(epoch))
    source = ColumnDataSource(dict(
        TPR=tpr,
        FPR=[1-cur_tnr for cur_tnr in tnr],
        epoch=epoch
    ))
    roc_plot.scatter(x='FPR', y='TPR',
                     color={'field': 'epoch',
                            'transform': col_mapper},
                     source=source)
    roc_plot.ray(x=0, y=0, length=1.42, angle=0.25*pi, color='grey')
    roc_plot.x_range = Range1d(0, 1)
    roc_plot.y_range = Range1d(0, 1)
    roc_plot.plot_width = 500
    roc_plot.plot_height = 500
    return roc_plot


def retrieve_read_properties(raw_read_dir, read_name):
    read_name_grep = re.search('(?<=/)[^/]+_strand', read_name).group()
    # Reconstruct full read name + path
    fast5_name = raw_read_dir + read_name_grep + '.fast5'
    try:
        hdf = h5py.File(fast5_name, 'r')
    except OSError:
        warnings.warn('Read %s not found in raw data, skipping read property retrieval.' % fast5_name, RuntimeWarning)
        return [nan for _ in range(5)]

    # Get metrics
    qscore = hdf['Analyses/Basecall_1D_000/Summary/basecall_1d_template'].attrs['mean_qscore']
    alignment = hdf['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment']
    alignment_metrics = [alignment.attrs[n] for n in ('num_deletions',
                                                      'num_insertions',
                                                      'num_mismatches',
                                                      'num_matches')]
    hdf.close()
    return [qscore] + alignment_metrics


def clean_classifications(y_hat, threshold=15):
    """
    Remove any detected event lasting shorter than threshold
    """
    y_hat_compressed = [[y_hat[0],0]]
    for yh in y_hat:
        if yh == y_hat_compressed[-1][0]:
            y_hat_compressed[-1][1] += 1
        else:
            y_hat_compressed.append([yh, 1])

    for yhci, yhc, in enumerate(y_hat_compressed):
        if yhc[0] != 0 and yhc[1] < threshold:
            y_hat_compressed[yhci][0] = 0
    return np.concatenate([np.repeat(yhc[0], yhc[1]) for yhc in y_hat_compressed])


def numeric_timestamp():
    return int(datetime.now().strftime('%H%M%S%f'))
