#!/usr/bin/python3

__all__ = [
    "plot_bandwidth_vs_size",
    "heatmaps_bandwidth",
    "get_image_html_tag",
]

import numpy
import matplotlib.pyplot as plt
from cycler import cycler
import io
import base64

custom_cycler = cycler(linestyle=["-", "--", ":", "-."]) * cycler("color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])

def plot_bandwidth_vs_size(df, size_name="size", prop_cycler=custom_cycler, **kwargs):
    """
    Creates a bandwidth-size plot. The "size" data are expected in the index of
    the dataframe, all other columns of the index are used for labels of the
    graph lines.

    :param df: a pandas.DataFrame instance
    :param size_name: name of the "size" column in the index
    :param prop_cycler:
        property cycler for the graph lines, see the documentation for details:
        https://matplotlib.org/stable/tutorials/intermediate/color_cycle.html
    :param kwargs:
        optional keyword arguments passed to matplotlib's errorbar function
    :returns: a tuple (fig, ax) as returned by plt.subplots()
    """
    # prepare the dataframe
    assert "bandwidth" in df.columns
    assert size_name in df.index.names
    df = df.reset_index(level=size_name).sort_index()

    # set default parameters for the plot
    kwargs.setdefault("capsize", 4)

    # plot the graph
    fig, ax = plt.subplots()
    ax.set_xlabel(size_name)
    ax.set_ylabel("bandwidth [GiB/s]")
    ax.set_prop_cycle(prop_cycler)
    for idx in df.index.unique():
        part = df.loc[idx]
        err = part["bandwidth"] * part["stddev/time"]
        ax.errorbar(part[size_name], part["bandwidth"], yerr=err, label=", ".join(idx), **kwargs)
    # see https://stackoverflow.com/a/43439132
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0.)

    return fig, ax

def heatmaps_bandwidth(df, x_name="columns", y_name="rows", *, cbar_kw=None, **kwargs):
    """
    Creates heatmaps two-dimensional data of bandwidth. The "size" data (i.e.
    x_name and y_name) are expected in the index of the dataframe, all other
    columns of the index are used to label the heatmaps. Heatmaps are generated
    using the Python generator interface for each unique tuple of dataframe
    index values.

    :param df: a pandas.DataFrame instance
    :param x_name: name of the column in the index to map along the x-axis
    :param y_name: name of the column in the index to map along the y-axis
    :param cbar_kw:
        optional dict of arguments passed to matplotlib's colorbar function
    :param kwargs:
        optional keyword arguments passed to matplotlib's imshow function
    :returns: a tuple (fig, ax) as returned by plt.subplots()
    """
    # prepare the dataframe
    assert "bandwidth" in df.columns
    assert x_name in df.index.names
    assert y_name in df.index.names
    df = df.reset_index(level=[x_name, y_name]).sort_index()

    if cbar_kw is None:
        cbar_kw = {}

    for idx in df.index.unique():
        # drop the index
        part = df.loc[idx].reset_index(drop=True)
        # get just the data we need
        part = part[[x_name, y_name, "bandwidth"]].set_index([y_name, x_name])
        # convert to a 2D array
        bandwidth = part.stack().unstack(level=x_name)
        # remove the column full of "bandwidth" from the index
        bandwidth = bandwidth.reset_index(level=1, drop=True)

        # figure setup
        fig, ax = plt.subplots()
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        label = ", ".join(idx)
        ax.set_title(f"{label} bandwidth [GiB/s]")

        # plot the heatmap and colorbar
        im = ax.imshow(bandwidth, interpolation=None, **kwargs)
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel("bandwidth", rotation=-90, va="bottom")

        # set ticks and their labels
        ax.set_xticks(numpy.arange(len(bandwidth.columns)))
        ax.set_yticks(numpy.arange(len(bandwidth.index)))
        ax.set_xticklabels(int(n) for n in bandwidth.columns)
        ax.set_yticklabels(int(n) for n in bandwidth.index)

        # rotate xtick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        yield fig, ax

def get_image_html_tag(fig, format="svg"):
    """
    Returns an HTML tag with embedded image data in the given format.

    :param fig: a matplotlib figure instance
    :param format: output image format (passed to fig.savefig)
    """
    stream = io.BytesIO()
    # bbox_inches: expand the canvas to include the legend that was put outside the plot
    # see https://stackoverflow.com/a/43439132
    fig.savefig(stream, format=format, bbox_inches="tight")
    data = stream.getvalue()

    if format == "svg":
        return data.decode("utf-8")
    data = base64.b64encode(data).decode("utf-8")
    return f"<img src=\"data:image/{format};base64,{data}\">"
