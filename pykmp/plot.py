import os

import graphviz
import numpy as np
import plotly
from typing_extensions import Literal


def to_graph(
    length: np.ndarray,
    path: np.ndarray,
    view: bool = False,
    cleanup: bool = True,
    **kwargs
):
    """
    Visualize paths using Graphviz.

    Args:
        path(np.ndarray): path path. N x 12(L1, ..., L6, N1, ..., N6) array.
        view(bool): open the graph in the default viewer. Default is False.
        cleanup(bool): clean up the generated files. Default is True.
        kwargs(dict): keyword arguments for graphviz.Digraph.
    """
    assert length.shape[0] == path.shape[0], (
        "length and path must have the same number of entries."
        f"got {length.shape[0]} and {path.shape[0]}."
    )
    _gX = lambda index: f'Group_{index:02X}'

    if 'format' not in kwargs:
        kwargs['format'] = 'png'

    # create graph
    dgraph = graphviz.Digraph(**kwargs)

    # set graph style
    dgraph.body.append(r"{rank=min; " + _gX(0) + r";}")
    for i, _path in enumerate(path):
        kwg = {'shape': 'box'}
        uniques = np.unique(_path[6:])
        uniques = uniques[:-1] if 255 in uniques else uniques
        if len(uniques) > 1:
            dgraph.body.append(
                "{rank=same; " + "; ".join([_gX(j) for j in uniques]) + ';}'
            )
            kwg['shape'] = 'ellipse'
        dgraph.node(_gX(i), **kwg)

    # create node
    for i, _path in enumerate(path):
        for nxt in np.unique(_path[6:]):
            if nxt == 255:
                break
            dgraph.edge(_gX(i), _gX(nxt), label=' {}'.format(length[i]))

    dgraph.render(cleanup=cleanup, view=view)


class _GraphvizSupport:
    """Add .to_graph method to class."""
    def to_graph(
        self,
        view: bool = False,
        cleanup: bool = True,
        **kwargs
    ):
        """
        Visualize paths using Graphviz.

        Args:
            view(bool): open the graph in the default viewer. Default is False.
            cleanup(bool): clean up the generated files. Default is True.
            kwargs(dict): keyword arguments for graphviz.Digraph.
        """
        path = np.concatenate([self.prev, self.next], axis=1)
        to_graph(self.length, path, view, cleanup, **kwargs)
