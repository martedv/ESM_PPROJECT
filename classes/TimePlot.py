# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import scale as mscale
from matplotlib.ticker import FuncFormatter
from classes.Plotting import Plotting
from classes.thesis_scales import LambertCylindricalLatitudeScale
from classes.thesis_colormap import ThesisPlotCols


class TimePlot(Plotting):
    def __init__(self, fig_scale=1.2, **kwargs: str):
        mscale.register_scale(LambertCylindricalLatitudeScale)
        fig_y = 3.4741 / fig_scale
        Plotting.__init__(self, fig_y=fig_y, **kwargs)

        self.ax = plt.subplot()
        self.ax.spines[["right", "top"]].set_visible(False)

        self.reset_cycle()

    def add_line(
        self,
        y: np.ndarray,
        time: np.ndarray,
        label: str = None,
        ax=None,
        **kwargs: str | int,
    ) -> None:
        if ax is None:
            self.ax.plot(time, y, linewidth=0.55, label=label, **kwargs)
        else:
            ax.plot(time, y, linewidth=0.55, label=label, **kwargs)

    def add_legend(self, **kwargs: str) -> None:
        self.ax.legend(frameon=False, **kwargs)

    def set_yformat(self, fmt: str) -> None:
        self.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt % x))

    def set_xformat(self, fmt: str) -> None:
        self.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt % x))

    def reset_cycle(self) -> None:
        self.ax.set_prop_cycle(color=ThesisPlotCols.optimal_scale())

    def add_horizontal(self, y: float = 0) -> None:
        self.ax.axhline(
            y,
            -90,
            90,
            color="gray",
            linestyle="--",
            alpha=0.5,
            zorder=0,
            linewidth=1,
        )


if __name__ == "__main__":

    ds2xpicadj = xr.load_dataset("../Data/Ocean/30Ma_2PIC_MOC_Ocean_years_1800-3000.nc")
    ds1xpic = xr.load_dataset("../Data/Ocean/30Ma_1PIC_Ocean_years_1800-4800.nc")

    ds2pic_time = np.array([])
    ds2pic_temp = np.array([])
    ds2pic_temp_l = np.array([])

    for file in os.listdir("../Data/Ocean/2PIC/"):
        ds = xr.load_dataset("../Data/Ocean/2PIC/%s" % file)
        ds2pic_time = np.append(ds2pic_time, ds["time"])
        ds2pic_temp = np.append(ds2pic_temp, ds["temp_u"])
        ds2pic_temp_l = np.append(ds2pic_temp_l, ds["temp_l"])

    tp = TimePlot()
    tp.add_line(ds2pic_temp, ds2pic_time, "2XPIC", zorder=10)

    tp.add_line(ds1xpic["temp_u"], ds1xpic["time"], "1XPIC")
    tp.add_line(ds2xpicadj["temp_u"], ds2xpicadj["time"], "2XPICadj")
    tp.add_horizontal(10)
    tp.ax.text(4200, 10.5, "Upper Ocean")
    tp.ax.text(4200, 9, "Lower Ocean")
    tp.reset_cycle()
    tp.add_line(ds2pic_temp_l, ds2pic_time, zorder=10)
    tp.add_line(ds1xpic["temp_l"], ds1xpic["time"])
    tp.add_line(ds2xpicadj["temp_l"], ds2xpicadj["time"])

    tp.set_yformat(r"\SI{%i}{\celsius}")
    tp.set_xformat(r"\SI{%i}{y}")

    tp.add_legend()
    tp.show()

    tp.savefig("diagnostic/ocean/temperature")

    # %%

    np.gradient(ds2xpicadj["temp_l"][-50:]).mean()
