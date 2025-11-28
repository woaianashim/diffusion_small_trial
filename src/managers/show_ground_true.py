import torch
import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display
from .base_manager import BaseInteractiveManager


class GroundTrue(BaseInteractiveManager):
    def __call__(self, show_masked_edges=False, t=0.5, scale=0.05):
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
        fig = go.FigureWidget(tree_data, layout=self.layout)

        scale_slider = widgets.FloatSlider(
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
            with fig.batch_update():
                vf_xy = self.prepare_edges(vf)
                fig.data[2].x = vf_xy["x"]
                fig.data[2].y = vf_xy["y"]
            self.regenerate_tree(fig)

        scale_slider.observe(update_scale, names="value")

        time_slider = widgets.FloatSlider(
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
            with fig.batch_update():
                vf_xy = self.prepare_edges(vf)
                fig.data[2].x = vf_xy["x"]
                fig.data[2].y = vf_xy["y"]
            self.regenerate_tree(fig)

        time_slider.observe(update_time, names="value")
        display(widgets.VBox([scale_slider, time_slider, fig]))
