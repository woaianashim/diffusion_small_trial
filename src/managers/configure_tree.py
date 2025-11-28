import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display
from .base_manager import BaseInteractiveManager
from dataclasses import dataclass


@dataclass
class ConfigureTree(BaseInteractiveManager):
    def __post_init__(self):
        self.fig = go.FigureWidget(
            self.tree_data(self.show_masked_edges), layout=self.layout
        )
        self.fan_slider = widgets.FloatSlider(
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
            self.regenerate_tree(self.fig)

        self.fan_slider.observe(update_fan, names="value")
        self.branch_slider = widgets.IntSlider(
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
            self.regenerate_tree(self.fig)

        self.branch_slider.observe(update_branching, names="value")

    def __call__(self):
        display(
            widgets.HBox(
                [widgets.VBox([self.fan_slider, self.branch_slider]), self.fig]
            )
        )
