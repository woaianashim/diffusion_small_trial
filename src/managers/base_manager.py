import numpy as np
import plotly.graph_objects as go
from types import SimpleNamespace
from dataclasses import dataclass, field
from ..base_density_model import BaseDensityModel


@dataclass
class BaseInteractiveManager:
    algo: BaseDensityModel
    state: SimpleNamespace = field(default_factory=SimpleNamespace)
    fig_size: int = 900
    show_masked_edges: bool = False

    @property
    def tree(self):
        return self.algo.data

    @property
    def cfg(self):
        return self.algo.cfg

    def tree_data(self, show_masked_edges=False):
        edges = self.tree.edges.detach().cpu().numpy()
        masked_edges = self.tree.masked_edges.detach().cpu().numpy()

        data = [
            go.Scatter(
                **self.prepare_edges(edges),
                mode="lines+markers",
                line=dict(width=2),
                marker=dict(size=6),
                hoverinfo="none",
                name="edges",
            ),
            go.Scatter(
                **(
                    self.prepare_edges(masked_edges)
                    if show_masked_edges
                    else {"x": [], "y": []}
                ),
                mode="lines+markers",
                line=dict(width=4, color="red"),
                marker=dict(size=10, color="red"),
                hoverinfo="none",
                name="masked_edge",
            ),
        ]
        return data

    def regenerate_tree(self, fig, show_masked_edges=False):
        self.tree.generate_tree()
        tree_data = self.tree_data(show_masked_edges)
        with fig.batch_update():
            fig.data[0].x = tree_data[0].x
            fig.data[0].y = tree_data[0].y
            fig.data[1].x = tree_data[1].x
            fig.data[1].y = tree_data[1].y

    @property
    def layout(self):
        edges_np = self.tree.edges.detach().cpu().numpy()
        pts = edges_np.reshape(-1, 2)  # shape (2N, 2)
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)

        xspan = float(xmax - xmin)
        yspan = float(ymax - ymin)
        span = max(xspan, yspan)
        if span == 0.0:
            span = 1.0  # avoid degenerate case for a single point

        # Add a bit of padding around the graph
        margin = 0.05 * span
        span = span + 2.0 * margin

        x_center = 0.5 * (xmin + xmax)
        y_center = 0.5 * (ymin + ymax)

        x_range = [x_center - span / 2.0, x_center + span / 2.0]
        y_range = [y_center - span / 2.0, y_center + span / 2.0]
        return go.Layout(
            title="Tree structure",
            hovermode="closest",
            xaxis=dict(
                title="X",
                range=x_range,
            ),
            yaxis=dict(
                title="Y",
                range=y_range,
                scaleanchor="x",
                scaleratio=1.0,
            ),
            width=self.fig_size,
            height=self.fig_size,
        )

    def prepare_edges(self, edges):
        x = edges[:, :, 0]
        y = edges[:, :, 1]

        def separate(x):
            x = np.concatenate([x, np.ones_like(x[:, :1])], axis=-1)
            x[:, -1] = np.nan
            x = x.reshape(-1)
            return x

        x = separate(x)
        y = separate(y)
        return {"x": x, "y": y}
