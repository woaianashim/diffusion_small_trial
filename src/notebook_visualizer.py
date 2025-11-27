import torch
from torchvision import io, utils
from torchvision.utils import draw_keypoints
import numpy as np
import torch
from typing import Sequence
import plotly.graph_objects as go
from functools import partial


def compute_pattern_matches(x: torch.Tensor, patterns) -> torch.Tensor:
    """
    x:        (N, ...) tensor
    patterns: (M, ...) list/np.array/torch.Tensor

    Returns:
        matches: (N, M) bool tensor, matches[n, m] = True
                 iff x[n] == patterns[m] elementwise.
    All consistency checks (ndim, trailing shape, device/dtype) are done here.
    """
    if x.ndim < 1:
        raise ValueError(f"x must have at least 1 dim (N,...), got {x.shape}")

    # Put patterns on same device/dtype as x
    patterns = torch.as_tensor(
        patterns,
        device=x.device,
        dtype=x.dtype,
    )

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
        return torch.zeros((N, 0), dtype=torch.bool, device=x.device)

    # Flatten all but the first dimension: (N, D) vs (M, D)
    x_flat = x.view(N, -1)
    pat_flat = patterns.view(M, -1)

    matches = (x_flat[:, None, :] == pat_flat[None, :, :]).all(dim=-1)  # (N, M)
    return matches


class Visualizer:
    def __init__(self,data,points,points_labels,highlight_color="red"):
        self.data=data
        self.points =points.cpu().numpy()
        self.points_labels=points_labels.cpu().numpy()
        self.edges=self.data.edges.cpu().numpy()      # (E, 2, 2)
        self.edge_labels_flat=self.get_edge_labels_flat()
        self.highlight_color=highlight_color

    def get_edge_labels_flat(self):
        edge_labels,_ = self.remove_masked_labels(self.data.all_labels,[self.data.masked_labels[0]])
        edge_labels_flat=edge_labels.view(edge_labels.shape[0], -1)
        edge_labels_flat=edge_labels_flat.cpu().numpy()
        return edge_labels_flat
    def get_colors(self,
            labels: torch.Tensor,
            patterns,
            pattern_colors,
            default_color: str = "blue",
    ):
        """
        labels:         (N, K) or (N, K, L)
        patterns:       (M, K) or (M, K, L) (same trailing shape as labels)
        pattern_colors: list of length M with color strings
        """
        N=labels.shape[0]
        matches = compute_pattern_matches(labels, patterns)  # (N, M)

        matches_np = matches.cpu().numpy()

        point_colors = np.full(N, default_color, dtype=object)

        for j, col in enumerate(pattern_colors):
            mask = matches_np[:, j]
            if mask.any():
                point_colors[mask] = col

        return point_colors
    def remove_masked_labels(self,
            edge_labels: torch.Tensor,
            masked_list: Sequence[torch.Tensor],
    ):
        """
        edge_labels: (E, K, L) tensor
        masked_list: list/seq of (K, L) tensors

        Returns:
            filtered:  (E', K, L) tensor with all entries from edge_labels
                       whose (K, L) block does NOT match any tensor in masked_list
            keep_mask: (E,) bool tensor, True where edge_labels[e] is kept
        """
        if edge_labels.ndim != 3:
            raise ValueError(f"edge_labels must be (E, K, L), got {edge_labels.shape}")

        E = edge_labels.shape[0]

        if not masked_list:
            keep_mask = torch.ones(E, dtype=torch.bool, device=edge_labels.device)
            return edge_labels, keep_mask

        # stack list -> (S, K, L); this will already throw if shapes are inconsistent
        masked_labels = torch.stack(list(masked_list), dim=0)  # (S, K, L)

        # All other compatibility checks happen inside compute_pattern_matches
        matches = compute_pattern_matches(edge_labels, masked_labels)  # (E, S)

        is_masked = matches.any(dim=1)  # (E,) bool
        keep_mask = ~is_masked

        filtered = edge_labels[keep_mask]  # (E', K, L)

        return filtered, keep_mask
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
    def get_figure(self,base_colors,out):
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
                    marker=dict(color=base_colors, size=6),
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

            # # # label of that edge in flattened space
            edge_label_flat = self.edge_labels_flat[edge_idx]            # (K_flat,)
            # # # compare to flattened point labels
            mask_t = (self.points_labels == edge_label_flat).all(axis=1)  # (N,) bool


            # # # recompute colors: start from base colors, only override highlighted points
            new_colors = base_colors.copy()
            new_colors[mask_t] = self.highlight_color

            # # # update highlighted edge coordinates
            hx = [x0[edge_idx], x1[edge_idx]]
            hy = [y0[edge_idx], y1[edge_idx]]

            with fig_tree.batch_update():
                points_trace.marker.color = new_colors
                highlight_trace.x = hx
                highlight_trace.y = hy

            with out:
                print(
                    f"Clicked near edge {edge_idx}, "
                    #f"{mask_t.sum()} points highlighted"
                )
        tree_trace.on_click(partial(handle_click, out=out))
        points_trace.on_click(partial(handle_click, out=out))
        return fig_tree
#%%
