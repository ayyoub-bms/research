import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable


def correlation_plot(correlation, classes=None, cmap='RdYlGn',
                     classes_color='black', **kwargs):

    fig, ax = plt.subplots()
    if classes is None:
        correl = correlation
    else:
        classes = classes.sort_values()
        name = classes.name
        class_counts = classes.reset_index().groupby(name).count()
        correl = correlation.loc[classes.index, classes.index]
        old = 0
        for i, c in enumerate(class_counts.SecurityID.cumsum()):
            x = [old, old]
            y = [old, c-1]
            z = [c-1, c-1]
            ax.plot(x, y, color=classes_color, lw=2.5)
            ax.plot(y, x, color=classes_color, lw=2.5)
            ax.plot(y, z, color=classes_color, lw=2.5)
            ax.plot(z, y, color=classes_color, lw=2.5)
            old = c

    m = ax.imshow(correl, cmap=cmap, **kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=.15)
    fig.colorbar(m, cax=cax)
    fig.tight_layout()

    return fig
