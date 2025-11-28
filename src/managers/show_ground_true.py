import torch
import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display
from .base_manager import BaseInteractiveManager


class GroundTrue(BaseInteractiveManager):
    def post_init(self, show_masked_edges=False, t=0.5, scale=0.05, **kwargs):
        tree_nodes = self.tree.nodes
        x, y = tree_nodes[:, 0], tree_nodes[:, 1]

        def get_grid(n):
            x_steps = torch.linspace(x.min(), x.max(), steps=n)
            y_steps = x_steps + y.min() - x.min()
            return torch.stack(torch.meshgrid(x_steps, y_steps), dim=-1).view(-1, 2)

        params = {"scale": scale, "t": t}
        grid = get_grid(50)
        gt_vf = self.algo.get_gt_vf(grid, t=torch.tensor(params["t"]))
        vf = torch.stack([grid, grid + gt_vf * params["scale"]], dim=1)
        vf_data = go.Scatter(
            **self.prepare_edges(vf),
            mode="lines",
            line=dict(width=1),
            hoverinfo="none",
            name="edges",
        )
        tree_data = self.tree_data(show_masked_edges)
        tree_data.append(vf_data)
        self.fig = go.FigureWidget(tree_data, layout=self.layout)

        self.scale_slider = widgets.FloatSlider(
            value=scale,
            min=0.01,
            max=0.2,
            step=0.01,
            description="Scale",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
        )

        def update_scale(change):
            params["scale"] = change["new"]
            vf = torch.stack([grid, grid + gt_vf * params["scale"]], dim=1)
            with self.fig.batch_update():
                vf_xy = self.prepare_edges(vf)
                self.fig.data[2].x = vf_xy["x"]
                self.fig.data[2].y = vf_xy["y"]
            self.regenerate_tree(self.fig)

        self.scale_slider.observe(update_scale, names="value")

        self.time_slider = widgets.FloatSlider(
            value=params["t"],
            min=0.01,
            max=0.99,
            step=0.01,
            description="Time",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
        )

        def update_time(change):
            params["t"] = change["new"]
            gt_vf = self.algo.get_gt_vf(grid, t=torch.tensor(params["t"]))
            vf = torch.stack([grid, grid + gt_vf * params["scale"]], dim=1)
            with self.fig.batch_update():
                vf_xy = self.prepare_edges(vf)
                self.fig.data[2].x = vf_xy["x"]
                self.fig.data[2].y = vf_xy["y"]
            self.regenerate_tree(self.fig)

        self.time_slider.observe(update_time, names="value")

    def __call__(self):
        display(widgets.VBox([self.scale_slider, self.time_slider, self.fig]))
