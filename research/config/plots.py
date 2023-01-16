import numpy as np
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt

green = '#34BE54'
blue = '#4482F4'
dark_gray = '#4B4B4B'
font = 'Stix'

_params = dict(
    ls=10,
    fs=11,
    ts=13,
    tp=13,
    lw=.9,
    mts=4,
    mats=7.5
)


def init(width=None, height=None, dpi=None):
    width = 7 if width is None else width
    height = width / 1.618 if height is None else height
    _get = _params.get
    # Font configuration
    plt.rc('font', size=_get('fs'), family=[font, 'Serif'])

    # Figure configuration
    plt.rc('figure', figsize=[width, height], dpi=dpi, autolayout=True)
    # plt.rc('text', usetex=True)
    plt.rc('lines', linewidth=_get('lw'))

    # Axes configuration
    plt.rc('axes',
           labelsize=_get('ls'),
           titlesize=_get('ts'),
           titlepad=_get('tp'))

    toggle_spines(top=False, right=False, bottom=True, left=True)
    toggle_grid(False)

    # Ticks configuration
    plt.rc('xtick', direction='in', labelsize=_get('ls'))
    plt.rc('ytick', direction='in', labelsize=_get('ls'))

    plt.rc('xtick.minor', visible=True, size=_get('mts'))
    plt.rc('ytick.minor', visible=True, size=_get('mts'))

    plt.rc('xtick.major', size=_get('mats'))
    plt.rc('ytick.major', size=_get('mats'))


def toggle_grid(toggle, minor=False, major=True):
    if minor and major:
        which = 'both'
    elif minor:
        which = 'minor'
    else:
        which = 'major'

    plt.rc('axes', grid=toggle)
    plt.rc('axes3d', grid=toggle)
    plt.rc('axes.grid', which=which)
    plt.rc('grid', color=dark_gray, alpha=.7, linestyle='--', linewidth=.5)


def toggle_spines(top=False, right=False, bottom=True, left=True):
    plt.rc('axes.spines', top=top, right=right, left=left, bottom=bottom)


def savefig(directory, filename, ext, fig=None):
    file = os.path.join(directory, f'{filename}.{ext}')
    if not os.path.isdir(directory):
        os.mkdir(directory)
    if fig is not None:
        fig.savefig(file)
    else:
        plt.savefig(file)
