# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import scale as mscale
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from classes.Plotting import Plotting
from classes.thesis_colormap import ThesisPlotCols
from classes.thesis_scales import LambertCylindricalLatitudeScale


class LatPlot(Plotting):
    xticks = [-90, -45, -30, -15, 0, 15, 30, 45, 90]

    def __init__(self, **kwargs: str):
        mscale.register_scale(LambertCylindricalLatitudeScale)
        fig_y = 3.4741 / 1.66
        Plotting.__init__(self, fig_y=fig_y, **kwargs)

        self.ax = plt.subplot()

        self.ax.set_xscale("lambert_cylindrical")

        self.ax.set_xlim(-90, 90)
        self.ax.spines[["right", "top"]].set_visible(False)

        self.reset_prop_cycle()

    def add_line(
        self, y: np.ndarray, lat: np.ndarray, label: str = None, **kwargs: str
    ) -> None:
        self.ax.plot(lat, y, linewidth=0.7, label=label, **kwargs)

    def add_horizontal(self, y: float = 0) -> None:
        self.ax.axhline(
            y,
            -90,
            90,
            color="gray",
            linestyle="--",
            alpha=0.5,
            zorder=0,
            linewidth=0.7,
        )

    def add_vertical(self, y: float = 0) -> None:
        self.ax.axvline(
            y,
            self.vmin,
            self.vmax,
            color="gray",
            linestyle="--",
            alpha=0.5,
            zorder=0,
            linewidth=0.7,
        )

    def set_ylim(self, min: float, max: float) -> None:
        self.vmin, self.vmax = min, max
        self.ax.set_ylim(min, max)

    def set_xlim(self, min: float, max: float) -> None:
        self.ax.set_xlim(min, max)

    def add_legend(self, **kwargs: str) -> object:
        return self.ax.legend(frameon=False, **kwargs)

    def set_yformat(self, fmt: str) -> None:
        self.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt % x))

    def reset_prop_cycle(self, ax: Axes = None) -> None:
        if ax is None:
            self.ax.set_prop_cycle(color=ThesisPlotCols.optimal_scale())
        else:
            ax.set_prop_cycle(color=ThesisPlotCols.optimal_scale())

    def smooth(self, x: np.ndarray, n: int = 4) -> np.ndarray:
        return np.convolve(x, np.ones(n) / n, mode="same")


if __name__ == "__main__":
    from thesis_classes.thesis_datasets import ThesisEBMs as eb

    eb1 = eb.eb_2d_pic2adj_c
    eb2 = eb.eb_2d_pic2_c

    tdc = eb1.temperature_decomp(eb2)

    lp = LatPlot()

    lp.add_line(tdc["Total"].mean(axis=(0, 2)) - tdc["Total"].mean(), eb.lat, r"Total")

    lp.add_line(
        tdc["albedo"].mean(axis=(0, 2)) - tdc["albedo"].mean(),
        eb.lat,
        r"$\alpha$ Albedo",
    )

    lp.add_line(
        tdc["emissivity"].mean(axis=(0, 2)) - tdc["emissivity"].mean(),
        eb.lat,
        label=r"$\epsilon$ Emissivity",
    )

    lp.add_line(
        tdc["heat"].mean(axis=(0, 2)),
        eb.lat,
        label=r"$H$ Heat",
    )

    lp.add_horizontal()
    lp.set_ylim(-2, 2)
    lp.add_legend()
    lp.show()

    lp.savefig("diagnostic/temperature/decomposition/1d/%s" % "decomp_2PICadj_2PIC")
