from .base_manager import BaseInteractiveManager
import torch
import numpy as np
import torch
from typing import Sequence
import plotly.graph_objects as go
from functools import partial
from dataclasses import dataclass


import numpy as np
import torch

def compute_pattern_matches(x, patterns):
    """
    x:        (N, ...) np.ndarray or torch.Tensor
    patterns: (M, ...) list/np.ndarray/torch.Tensor

    Returns:
        matches: (N, M) bool array/tensor, matches[n, m] = True
                 iff x[n] == patterns[m] elementwise.

    All consistency checks (ndim, trailing shape, device/dtype) are done here.
    The backend (NumPy vs Torch) is inferred from `x` and preserved.
    """
    # --- infer backend ---
    is_torch = torch.is_tensor(x)
    if not (is_torch or isinstance(x, np.ndarray)):
        raise TypeError(
            f"x must be torch.Tensor or np.ndarray, got {type(x)}"
        )

    # x: (N, ...)
    if x.ndim < 1:
        raise ValueError(f"x must have at least 1 dim (N,...), got {x.shape}")

    # --- convert patterns to same backend as x ---
    if is_torch:
        patterns = torch.as_tensor(
            patterns,
            device=x.device,
            dtype=x.dtype,
        )
    else:
        x = np.asarray(x)
        patterns = np.asarray(patterns, dtype=x.dtype)

    # --- shape checks ---
    if patterns.ndim != x.ndim:
        raise ValueError(
            f"patterns.ndim={patterns.ndim} must equal x.ndim={x.ndim}; "
            f"got x.shape={x.shape}, patterns.shape={patterns.shape}"
        )

    if patterns.shape[1:] != x.shape[1:]:
        raise ValueError(
            f"Shape mismatch in trailing dims: x.shape[1:]={x.shape[1:]}, "
            f"patterns.shape[1:]={patterns.shape[1:]}"
        )

    N = x.shape[0]
    M = patterns.shape[0]

    if M == 0:
        # No patterns: return an empty matches matrix
        if is_torch:
            return torch.zeros((N, 0), dtype=torch.bool, device=x.device)
        else:
            return np.zeros((N, 0), dtype=bool)

    # --- flatten all but the first dimension: (N, D) vs (M, D) ---
    x_flat = x.reshape(N, -1)
    pat_flat = patterns.reshape(M, -1)

    # --- broadcast and compare: (N, M, D) -> (N, M) ---
    if is_torch:
        matches = (x_flat[:, None, :] == pat_flat[None, :, :]).all(dim=-1)
    else:
        matches = (x_flat[:, None, :] == pat_flat[None, :, :]).all(axis=-1)

    return matches



@dataclass
class PointCloudVisualizer(BaseInteractiveManager):

    def __post_init__(self,points=None,points_labels=None,edges=None,edge_colors=None, highlight_color="red",default_color="blue"):
        if points is None or points_labels is None:
            self.points,self.points_labels = self.default_points_generators(points_labels)
        else:
            self.points = points.cpu().numpy()
            self.points_labels=points_labels.cpu().numpy()

        self.edges=self.algo.data.edges.cpu().numpy()
        self.edge_labels_flat=self.algo.data.labels.view(self.algo.data.labels.shape[0],-1).cpu().numpy()
        self.base_colors=self.get_colors(edges,
                                        edge_colors,default_color)
        self.highlight_color=highlight_color


    def default_points_generators(self,points_labels):
        point_dict=self.algo.data.sample_batch(10000)
        return point_dict["gt"].cpu().numpy(), point_dict["label"].cpu().numpy()

    def get_colors(self,
                        patterns,
                        pattern_colors,
                        default_color: str = "blue",
                        ):
        if patterns is None or pattern_colors is None:
            patterns = [
                self.algo.data.masked_labels[0].view(-1).cpu().numpy(),
            ]
            pattern_colors=["green"]

        N=self.points_labels.shape[0]
        matches = compute_pattern_matches(self.points_labels, patterns)  # (N, M)
        if torch.is_tensor(matches):
            matches_np = matches.cpu().numpy()
        else:
            matches_np=matches
        point_colors = np.full(N, default_color, dtype=object)

        for j, col in enumerate(pattern_colors):
            mask = matches_np[:, j]
            if mask.any():
                point_colors[mask] = col

        return point_colors
    def find_nearest_edge(self,cx, cy):

        x0 = self.edges[:, 0, 0]
        y0 = self.edges[:, 0, 1]
        x1 = self.edges[:, 1, 0]
        y1 = self.edges[:, 1, 1]
        dx = x1 - x0
        dy = y1 - y0
        denom = dx * dx + dy * dy
        denom = np.where(denom == 0, 1e-12, denom)

        t = ((cx - x0) * dx + (cy - y0) * dy) / denom
        t = np.clip(t, 0.0, 1.0)

        proj_x = x0 + t * dx
        proj_y = y0 + t * dy

        dist2 = (proj_x - cx) ** 2 + (proj_y - cy) ** 2
        idx = dist2.argmin()
        return idx
    def get_figure(self,out):

        x0 = self.edges[:, 0, 0]
        y0 = self.edges[:, 0, 1]
        x1 = self.edges[:, 1, 0]
        y1 = self.edges[:, 1, 1]
        edge_x = []
        edge_y = []
        for e in self.edges:
            edge_x += [e[0, 0], e[1, 0], None]
            edge_y += [e[0, 1], e[1, 1], None]

        fig_tree = go.FigureWidget(
            data=[
                # 0: points
                go.Scatter(
                    x=self.points[:, 0],
                    y=self.points[:, 1],
                    mode="markers",
                    name="points",
                    marker=dict(color=self.base_colors, size=6),
                ),
                # 1: full tree
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    name="tree",
                    hoverinfo="none",
                    line=dict(width=1),
                ),
                # 2: highlighted edge
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    name="selected_edge",
                    line=dict(width=4),
                ),
            ]
        )

        fig_tree.update_layout(
            xaxis=dict(scaleanchor="y", scaleratio=1),
            showlegend=False,
            width=1100,
            height=1100
        )
        points_trace    = fig_tree.data[0]
        tree_trace      = fig_tree.data[1]
        highlight_trace = fig_tree.data[2]

        def handle_click(trace, points_clicked, selector,out):
            if not points_clicked.xs:
                return

            cx = points_clicked.xs[0]
            cy = points_clicked.ys[0]

            edge_idx = self.find_nearest_edge(cx, cy)

            # # # # label of that edge in flattened space
            edge_label_flat = self.edge_labels_flat[edge_idx]            # (K_flat,)
            # # # # compare to flattened point labels
            mask_t = (self.points_labels == edge_label_flat).all(axis=1)  # (N,) bool
            #
            #
            # # # # recompute colors: start from base colors, only override highlighted points
            new_colors = self.base_colors.copy()
            new_colors[mask_t] = self.highlight_color
            #
            # # # # update highlighted edge coordinates
            hx = [x0[edge_idx], x1[edge_idx]]
            hy = [y0[edge_idx], y1[edge_idx]]
            #
            with fig_tree.batch_update():
                 points_trace.marker.color = new_colors
                 highlight_trace.x = hx
                 highlight_trace.y = hy

            with out:
                print(
                    f"Clicked near edge {edge_idx}, "
                    f"{mask_t.sum()} points highlighted"
                )
        tree_trace.on_click(partial(handle_click, out=out))
        points_trace.on_click(partial(handle_click, out=out))
        return fig_tree

