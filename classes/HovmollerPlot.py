import calendar
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import scale as mscale
from classes.Plotting import Plotting
from classes.thesis_scales import LambertCylindricalLatitudeScale


class HovmollerPlot(Plotting):
    months: list = []
    cb_ticks = None
    extend = None

    def __init__(
        self,
        values: np.ndarray,
        lat: np.ndarray,
        levels: int = 10,
        extend: str = "max",
        mode: Literal["Equator", "Full"] = "Equator",
        **kwargs: str,  # noqa
    ) -> None:
        fig_y = 3.4741 / 1.1
        Plotting.__init__(self, fig_y=fig_y, **kwargs)

        self.values = values
        self.lat = lat
        self.levels = levels
        self.extend = extend

        self.vmin = self.values.min()
        self.vmax = self.values.max()

        self.months = self._get_calendar_months()
        self.ax = self.fig.add_subplot(2, 1, 1)
        mscale.register_scale(LambertCylindricalLatitudeScale)
        self.ax.set_yscale("lambert_cylindrical")
        self.ax.set_ylim(-90, 90)

        if mode == "Equator":
            self.ax.set_ylim(-30, 30)

    def _get_calendar_months(self) -> list:
        return [calendar.month_abbr[i] for i in range(1, 13)]

    def _render_original(self) -> None:

        self.render = self.ax.contourf(
            self.months,
            self.lat,
            self.values.transpose(),
            levels=np.linspace(self.vmin, self.vmax, self.levels),
            extend=self.extend,
            cmap=self.cmap,
        )
        self.render_contours = self.ax.contour(
            self.months,
            self.lat,
            self.values.transpose(),
            levels=np.linspace(self.vmin, self.vmax, self.levels),
            linewidths=0.2,
            colors=["black"],
        )

    def add_grid(self) -> None:
        self.ax.xaxis.tick_top()
        plt.grid(linewidth=0.3)
        plt.tick_params(pad=0.2)

    def set_zlim(self, vmin: float, vmax: float) -> None:
        self.vmin, self.vmax = vmin, vmax
