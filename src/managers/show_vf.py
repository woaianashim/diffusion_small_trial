import torch
import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display
from .base_manager import BaseInteractiveManager
from enum import Enum
from dataclasses import dataclass


class Mode(Enum):
    GT = "GT"
    PRED = "PRED"
    DIFF = "DIFF"


@dataclass
class VectorField(BaseInteractiveManager):
    t: float = 0.5
    scale: float = 0.3
    grid_size: int = 50
    mode: Mode = Mode.GT
    vector: bool = True
    title: str = "Vector Field"

    def __post_init__(self):
        tree_nodes = self.tree.nodes
        x, y = tree_nodes[:, 0], tree_nodes[:, 1]

        def get_grid(n):
            x_steps = torch.linspace(x.min(), x.max(), steps=n)
            y_steps = x_steps + y.min() - x.min()
            return torch.stack(torch.meshgrid(x_steps, y_steps), dim=-1).view(-1, 2)

        grid = get_grid(self.grid_size)

        self.state.vf = self.get_vector_field(grid)
        self.state.vf /= self.state.vf.norm(dim=-1).max()
        vf = torch.stack([grid, grid + self.state.vf * self.scale], dim=1)
        vf_data = go.Scatter(
            **self.prepare_edges(vf),
            mode="lines",
            line=dict(width=1),
            hoverinfo="none",
            name="edges",
        )
        tree_data = self.tree_data(self.show_masked_edges)
        tree_data.append(vf_data)
        self.fig = go.FigureWidget(tree_data, layout=self.layout)

        def update_figure(update_data=False):
            if update_data:
                self.fig.layout.title = "Processing..."
                self.state.vf = self.get_vector_field(grid)
                self.state.vf /= self.state.vf.norm(dim=-1).max()
                self.fig.layout.title = self.title
            vf = torch.stack([grid, grid + self.state.vf * self.scale], dim=1)
            with self.fig.batch_update():
                vf_xy = self.prepare_edges(vf)
                self.fig.data[2].x = vf_xy["x"]
                self.fig.data[2].y = vf_xy["y"]
            self.regenerate_tree(self.fig)

        self.scale_slider = widgets.FloatSlider(
            value=self.scale,
            min=0.01,
            max=0.7,
            step=0.01,
            description="Scale",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
        )

        def update_scale(change):
            self.scale = change["new"]
            update_figure()

        self.scale_slider.observe(update_scale, names="value")

        self.time_slider = widgets.FloatSlider(
            value=self.t,
            min=0.001,
            max=0.99,
            step=0.001,
            description="Time",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".3f",
        )

        def update_time(change):
            self.t = change["new"]
            update_figure(True)

        self.time_slider.observe(update_time, names="value")
        self.mode_toggler = widgets.ToggleButtons(
            options=[Mode.GT, Mode.PRED, Mode.DIFF],
            description="Mode:",
            disabled=False,
            button_style="",
        )

        def update_mode(change):
            self.mode = change["new"]
            update_figure(True)

        self.mode_toggler.observe(update_mode, names="value")

    def __call__(self):
        display(
            widgets.VBox(
                [self.mode_toggler, self.scale_slider, self.time_slider, self.fig]
            )
        )

    def get_vector_field(self, grid):
        if self.mode == Mode.GT:
            return self.algo.get_gt_vf(grid, t=torch.tensor(self.t))
        elif self.mode == Mode.DIFF:
            gt = self.algo.get_gt_vf(grid, t=torch.tensor(self.t))
            with torch.no_grad():
                pred = self.algo.get_vector_field(
                    grid.to(self.cfg.device), torch.tensor(self.t).to(self.cfg.device)
                ).cpu()
            return gt - pred
        else:
            with torch.no_grad():
                pred = self.algo.get_vector_field(
                    grid.to(self.cfg.device), torch.tensor(self.t).to(self.cfg.device)
                ).cpu()
            return pred
