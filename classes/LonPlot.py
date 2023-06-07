# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from classes.Plotting import Plotting


class LonPlot(Plotting):
    lon_ticks = [-180, -135, -90, -45, 0, 45, 90, 135, 180]

    def __init__(self, **kwargs: str):
        fig_y = 3.4741 / 1.66
        Plotting.__init__(self, fig_y=fig_y, **kwargs)

        self.ax = plt.subplot()

        self.ax.set_xlim(-180, 180)
        self.ax.xaxis.set_ticks(self.lon_ticks)
        self.ax.spines[["right", "top"]].set_visible(False)

        self.ax.set_prop_cycle(color=ThesisPlotCols.optimal_scale())

    def add_line(self, y: np.ndarray, lat: np.ndarray, label: str = None) -> None:
        self.ax.plot(lat - 180, y, linewidth=0.7, label=label)

    def add_horizontal(self, y: float = 0) -> None:
        self.ax.axhline(
            y,
            -180,
            180,
            color="gray",
            linestyle="--",
            alpha=0.5,
            zorder=0,
            linewidth=0.7,
        )

    def set_ylim(self, min: float, max: float) -> None:
        self.ax.set_ylim(min, max)

    def set_xlim(self, min: float, max: float) -> None:
        self.ax.set_xlim(min, max)

    def add_legend(self, **kwargs: str) -> None:
        self.ax.legend(frameon=False, **kwargs)

    def set_yformat(self, fmt: str) -> None:
        self.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt % x))
