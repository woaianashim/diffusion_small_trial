import torch
import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display
from src.managers.base_manager import BaseInteractiveManager
from dataclasses import dataclass


@dataclass
class Tracks(BaseInteractiveManager):
    t: float = 0.5
    scale: float = 0.3
    grid_size: int = 50
    title: str = "Sampling tracks"

    def __post_init__(self):
        self.original_noise = []

        tree_data = self.tree_data(self.show_masked_edges)
        tree_data.append(
            go.Scatter(
                x=[], y=[], name="GT", mode="lines+markers", line=dict(color="green")
            )
        )  # Will be replaced by GT track
        tree_data.append(
            go.Scatter(
                x=[],
                y=[],
                name="Predicted",
                mode="lines+markers",
                line=dict(color="orange"),
            )
        )  # Will be replaced by Predicted track

        x_range, y_range = self.fig_ranges
        xs = torch.linspace(x_range[0], x_range[1], self.grid_size)
        ys = torch.linspace(y_range[0], y_range[1], self.grid_size)
        grid_x, grid_y = torch.meshgrid(xs, ys)
        click_x = grid_x.reshape(-1).tolist()
        click_y = grid_y.reshape(-1).tolist()
        click_scatter = go.Scatter(
            x=click_x,
            y=click_y,
            mode="markers",
            marker=dict(size=40, opacity=0),
            showlegend=False,
            hoverinfo="none",
            name="_click_capture",
        )

        tree_data.append(click_scatter)
        self.fig = go.FigureWidget(tree_data, layout=self.layout)

        def update_figure():
            if len(self.original_noise) == 0:
                return
            self.fig.layout.title = "Processing..."
            initial_noise = torch.stack(self.original_noise, dim=0)  # Bx2
            _, gt_steps = self.algo.sample(
                len(self.original_noise), initial_noise=initial_noise, from_gt=True
            )
            _, pred_steps = self.algo.sample(
                len(self.original_noise), initial_noise=initial_noise, from_gt=False
            )
            self.fig.layout.title = self.title
            with self.fig.batch_update():
                gt_steps = self.prepare_edges(
                    torch.stack(gt_steps, dim=0).detach().cpu().transpose(1, 0).numpy()
                )  # TxBx2
                self.fig.data[2].x = gt_steps["x"]
                self.fig.data[2].y = gt_steps["y"]
                pred_steps = self.prepare_edges(
                    torch.stack(pred_steps, dim=0)
                    .detach()
                    .cpu()
                    .transpose(1, 0)
                    .numpy()
                )  # TxBx2
                self.fig.data[3].x = pred_steps["x"]
                self.fig.data[3].y = pred_steps["y"]
            self.regenerate_tree(self.fig)

        def handle_click(trace, points, selector):
            for idx in points.point_inds:
                x = trace.x[idx]
                y = trace.y[idx]
                self.original_noise.append(
                    torch.tensor([float(x), float(y)], dtype=torch.float32)
                )
                update_figure()

        self.fig.data[-1].on_click(handle_click)

        def on_submit_clicked(_):
            update_figure()

        def on_clear_clicked(_):
            self.original_noise.clear()
            with self.fig.batch_update():
                self.fig.data[2].x = []
                self.fig.data[2].y = []
                self.fig.data[3].x = []
                self.fig.data[3].y = []
            self.fig.layout.title = self.title

        self.submit_button = widgets.Button(description="Submit")
        self.clear_button = widgets.Button(description="Clear")

        self.submit_button.on_click(on_submit_clicked)
        self.clear_button.on_click(on_clear_clicked)

    def __call__(self):
        display(widgets.VBox([self.submit_button, self.clear_button, self.fig]))
