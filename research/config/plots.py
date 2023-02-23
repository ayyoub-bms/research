import numpy as np
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt

dgray = '#4B4B4B'
font = 'Sans'

_params = dict(
    ls=10,
    fs=11,
    ts=13,
    tp=13,
    lw=1.5,
    mts=4,
    mats=7.5
)


def init(width=None, height=None, dpi=120):
    width = width or 7
    height = height or width / 1.618
    _get = _params.get
    # Font configuration
    plt.rc('font', size=_get('fs'), family=[font, 'Serif'])

    # Figure configuration
    plt.rc('figure', figsize=[width, height], dpi=dpi)
    plt.rc('text', usetex=True)
    plt.rc('lines', linewidth=_get('lw'))

    # Axes configuration
    plt.rc('axes',
           labelsize=_get('ls'),
           titlesize=_get('ts'),
           titlepad=_get('tp'))

    toggle_spines(top=False, right=False, bottom=True, left=True)
    toggle_grid()

    # Ticks configuration
    plt.rc('xtick', direction='in', labelsize=_get('ls'))
    plt.rc('ytick', direction='in', labelsize=_get('ls'))
    plt.rc('xtick.minor', visible=True, size=_get('mts'))
    plt.rc('ytick.minor', visible=True, size=_get('mts'))

    plt.rc('xtick.major', size=_get('mats'))
    plt.rc('ytick.major', size=_get('mats'))


def toggle_grid(toggle=True, minor=False, major=True):
    if minor and major:
        which = 'both'
    elif minor:
        which = 'minor'
    else:
        which = 'major'

    plt.rc('axes', grid=toggle)
    plt.rc('axes3d', grid=toggle)
    plt.rc('axes.grid', which=which)
    plt.rc('grid', color=dgray, alpha=.7, linestyle='--', linewidth=.5)


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


def keep_axes(axes, n):
    """ Keep only n axes from `axes`"""
    axes = axes.flat
    for ax in axes[n:]:
        ax.remove()
    return axes[:n]
