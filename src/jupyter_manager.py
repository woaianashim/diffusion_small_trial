import os
import torch
import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra import compose, initialize_config_dir
from IPython.display import display, clear_output


class InteractiveManager:
    def __init__(self, fig_size=900):
        self.fig_size = fig_size
        GlobalHydra.instance().clear()
        config_dir = os.path.join(os.getcwd(), "config")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="conf",
                return_hydra_config=True,
                overrides=[
                    "+algo=diffusion",
                ],
            )
            HydraConfig.instance().set_config(cfg)
            self.cfg = cfg

    @property
    def tree(self):
        return self.algo.data

    def setup(self, algo="diffusion"):

        if algo == "diffusion":
            from src.diffusion import Diffusion

            self.algo = Diffusion(self.cfg)
        elif algo == "fm":
            from src.fm import FlowMatching

            self.algo = FlowMatching(self.cfg)
        else:
            raise AttributeError("Algo should be diffusion or fm")

    def show_tree(self):
        fig = go.FigureWidget(self.tree_data(), layout=self.layout)
        fan_slider = widgets.FloatSlider(
            value=1.0,
            min=0.3,
            max=2.0,
            step=0.01,
            description="Fan",
            disabled=False,
            continuous_update=True,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
        )

        def update_fan(change):
            self.tree.cfg.tree.fan = change["new"]
            self.regenerate_tree(fig)

        fan_slider.observe(update_fan, names="value")
        branch_slider = widgets.IntSlider(
            value=3,
            min=2,
            max=4,
            step=1,
            description="Branching",
            disabled=False,
            continuous_update=True,
            orientation="horizontal",
            readout=True,
        )

        def update_branching(change):
            self.tree.cfg.tree.branching = change["new"]
            self.regenerate_tree(fig)

        branch_slider.observe(update_branching, names="value")
        display(widgets.HBox([widgets.VBox([fan_slider, branch_slider]), fig]))

    def select_edge(self, show_masked_edges=False):
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
            self.tree.label_masks[idx] = ~self.tree.label_masks[idx]
            tree_data = self.tree_data(show_masked_edges)
            with fig.batch_update():
                fig.data[0].x = tree_data[0].x
                fig.data[0].y = tree_data[0].y
                fig.data[1].x = tree_data[1].x
                fig.data[1].y = tree_data[1].y
            info_label.value = f"Selected edge #{edge_id_by_index[idx]}"

            with out:
                clear_output(wait=True)

        edge_buttons.on_click(handle_click)

    def show_ground_true_vector_field(self, t=0.5, scale=0.05):
        tree_nodes = self.tree.nodes
        x, y = tree_nodes[:, 0], tree_nodes[:, 1]

        def get_grid(n):
            x_steps = torch.linspace(x.min(), x.max(), steps=n)
            y_steps = x_steps + y.min() - x.min()
            return torch.stack(torch.meshgrid(x_steps, y_steps), dim=-1).view(-1, 2)

        grid = get_grid(50)
        gt_vf = self.algo.get_gt_vf(grid, t=torch.tensor(t))
        vf = torch.stack([grid, grid + gt_vf * scale], dim=1)
        vf_data = go.Scatter(
            **self.prepare_edges(vf),
            mode="lines+markers",
            line=dict(width=2),
            marker=dict(size=6),
            hoverinfo="none",
            name="edges",
        )
        tree_data = self.tree_data()
        tree_data.append(vf_data)
        info_label = widgets.Label(value="GT vector field")
        out = widgets.Output()
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
            new_scale = change["new"]
            vf = torch.stack([grid, grid + gt_vf * new_scale], dim=1)
            with fig.batch_update():
                vf_xy = self.prepare_edges(vf)
                fig.data[2].x = vf_xy["x"]
                fig.data[2].y = vf_xy["y"]
            self.regenerate_tree(fig)

        scale_slider.observe(update_scale, names="value")
        display(widgets.VBox([scale_slider, fig]))

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
