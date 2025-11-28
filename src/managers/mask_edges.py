import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display, clear_output
import numpy as np
from .base_manager import BaseInteractiveManager


class MaskEdges(BaseInteractiveManager):
    def __call__(self, show_masked_edges=False):
        edges_all = self.tree.all_edges.detach().cpu().numpy()
        labels = self.tree.all_labels.detach().cpu().numpy()
        labels = labels.reshape(
            -1, self.algo.data.cfg.tree.depth, self.algo.data.cfg.tree.branching + 1
        ).argmax(-1)
        centers = edges_all.mean(1)
        edge_id_by_index = np.arange(centers.shape[0])

        tree_data = self.tree_data(show_masked_edges)
        tree_data.append(
            go.Scatter(
                x=centers[..., 0],
                y=centers[..., 1],
                mode="markers",
                marker=dict(size=5, color="orange"),
                name="centers",
                hoverinfo="text",
                text=[
                    f"edge {labels[eid]}" if eid >= 0 else ""
                    for eid in edge_id_by_index
                ],
            ),
        )
        info_label = widgets.Label(value="Select edge")
        out = widgets.Output()
        fig = go.FigureWidget(tree_data, layout=self.layout)
        display(info_label, fig, out)

        edge_buttons = fig.data[-1]

        def handle_click(trace, edges, state):
            if not edges.point_inds:
                return

            idx = edges.point_inds[0]
            assert self.tree.label_masks is not None
            self.tree.label_masks[idx] = ~self.tree.label_masks[idx]
            self.regenerate_tree(fig, show_masked_edges)
            info_label.value = f"Selected edge #{edge_id_by_index[idx]}"

            with out:
                clear_output(wait=True)

        edge_buttons.on_click(handle_click)
