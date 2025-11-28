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


@dataclass
class Sampler(BaseInteractiveManager):
    t: float = 0.5
    mode: Mode = Mode.PRED
    title: str = "Sample via model"
    N: int = 1000
    num_steps: int = 50
    point_size: int = 10

    def __post_init__(self):
        tree_data = self.tree_data(self.show_masked_edges)
        self.algo.cfg.sampler.num_steps = self.num_steps
        _, steps = self.algo.sample(self.N)
        steps = torch.stack(steps, dim=0).detach().cpu().numpy()  # TxBx2
        frames_data = [
            [
                go.Scatter(
                    x=step[:, 0],
                    y=step[:, 1],
                    mode="markers",
                    opacity=0.5,
                    marker=dict(color="orange", size=self.point_size),
                )
            ]
            for step in steps
        ]
        frames = [
            go.Frame(data=frame_data, traces=[2], name=f"frame_{i}")
            for i, frame_data in enumerate(frames_data)
        ]
        layout = self.layout
        layout.updatemenus = [
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": self.num_steps, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                ],
            }
        ]
        slider_steps = [
            {
                "args": [
                    [f"frame_{k}"],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
                "label": str(k),
                "method": "animate",
            }
            for k in range(len(frames))
        ]

        layout.sliders = [
            dict(
                active=0,
                currentvalue={"prefix": "Step: "},
                pad={"t": 10, "b": 50},
                len=0.9,
                x=0.1,
                y=1.1,
                xanchor="left",
                steps=slider_steps,
            )
        ]
        tree_data.append(frames_data[0][0])
        self.fig = go.Figure(data=tree_data, layout=layout, frames=frames)

    def __call__(self):
        self.fig.show()
