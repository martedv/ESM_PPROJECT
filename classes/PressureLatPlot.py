# %%
import os

if "first_run" not in globals():

    os.chdir("../")
    first_run = True

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import scale as mscale
from matplotlib.ticker import FuncFormatter
from classes.Plotting import Plotting
from classes.thesis_scales import LambertCylindricalLatitudeScale


class PressureLatPlot(Plotting):
    xticks = [-90, -45, -30, -15, 0, 15, 30, 45, 90]
    levels = 10
    levs: np.ndarray | None = None
    yticks = [1000, 800, 600, 400, 200]

    def __init__(
        self,
        z: np.ndarray,
        lat: np.ndarray,
        pl: np.ndarray,
        levels: int = 10,
        **kwargs: str,
    ):
        mscale.register_scale(LambertCylindricalLatitudeScale)
        fig_y = 3.4741 / 1.66
        Plotting.__init__(self, fig_y=fig_y, **kwargs)

        self.ax = plt.subplot()

        self.ax.set_xscale("lambert_cylindrical")

        self.lat, self.pl, self.z = lat, pl, z

        self.vmax, self.vmin = z.max(), z.min()
        self.levels = levels
        self.ax.set_xticks(self.xticks)
        self.set_yformat("%i hPa")

    def set_yformat(self, fmt: str) -> None:
        self.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: fmt % x))

    def _render_original(self) -> None:
        self.levs = (
            np.linspace(self.vmin, self.vmax, self.levels)
            if self.levs is None
            else self.levs
        )

        self.render = self.ax.contourf(
            self.lat,
            self.pl,
            self.z,
            levels=self.levs,
            cmap=self.cmap,
            extend="both",
        )
        self.ax.set_ylim(self.ax.get_ylim()[::-1])
        self.ax.set_yticks(self.yticks)
        self.ax.set_xlim(-45, 45)
        self.ax.xaxis.tick_top()

    def set_zlim(self, vmin: float, vmax: float) -> None:
        self.vmin, self.vmax = vmin, vmax

    def add_overlay(self, z_l: np.ndarray, lat: np.ndarray, pl: np.ndarray) -> None:

        self.ax.contour(lat, pl, z_l, levels=self.levs, cmap=self.cmap)
        self.ax.contour(
            lat,
            pl,
            z_l,
            levels=self.levs,
            cmap=self.cmap,
            linewidths=0.5,
            alpha=0.7,
            zorder=2,
        )
        self.ax.contour(
            lat,
            pl,
            z_l,
            levels=self.levs,
            colors="black",
            linewidths=0.15,
            zorder=3,
        )


if __name__ == "__main__":
    import cmocean
    from thesis_classes.thesis_datasets import ThesisEBMs as eb
    from scipy.interpolate import griddata

    def p_plot(
        hadc: np.ndarray,
        diff: bool = False,
        overlay: np.ndarray = None,
        name: str = None,
        showcb: bool = True,
        showquiv: bool = True,
    ) -> None:
        pl = PressureLatPlot(hadc, eb.lat, eb.eb_2d_pic2_c.lev, 30)
        heights = eb.eb_2d_pic2_c.dataset.get_variable("Z3").mean(axis=(0, 2, 3))

        v = np.gradient(hadc, 210685, axis=1) * 100
        w = -np.gradient(hadc, heights, axis=0)

        if diff:
            pl.set_zlim(-50, 50)
            pl.cb_ticks = np.arange(-50, 51, 25)
        else:
            pl.set_zlim(-200, 200)
            pl.cb_ticks = np.arange(-200, 201, 50)
        if diff:
            pl.set_thesis_cmap("itcz_delta")
        else:
            pl.set_cmap(cmocean.cm.balance)
        pl.show()

        if overlay is not None:
            pass
            # pl.add_overlay(overlay, eb.lat, eb.eb_2d_pic2_c.lev)

        mask = np.sqrt(w**2 + v**2) < 0.002

        w = np.ma.masked_array(w, mask)
        v = np.ma.masked_array(v, mask)
        if showquiv:
            pl.ax.quiver(
                eb.lat,
                eb.eb_2d_pic2_c.lev,
                w,
                v,
                scale=1,
                headwidth=3,
                width=0.002,
            )
        if showcb:
            pl.add_colorbar("%i")
            pl.cbar.set_label("Streamfunction \SI{1e9}{\kg\per\second}")

        if name is not None:
            pl.savefig(name)

    phi_pic2 = eb.eb_2d_pic2_c.hadley_cell()

    phi_pic2adj = eb.eb_2d_pic2adj_c.hadley_cell()
    phi_pic1 = eb.eb_2d_pic1_c.hadley_cell()

    p_plot(
        phi_pic2[[11, 0, 1]].mean(axis=0),
        overlay=phi_pic2[[11, 0, 1]].mean(axis=0),
        name="diagnostic/Hadley/total/2XPIC_Hadley_SF_DJF",
    )

    p_plot(
        phi_pic2adj[[11, 0, 1]].mean(axis=0),
        overlay=phi_pic2adj[[11, 0, 1]].mean(axis=0),
        name="diagnostic/Hadley/total/2XPICadj_Hadley_SF_DJF",
    )

    p_plot(
        phi_pic1[[11, 0, 1]].mean(axis=0),
        overlay=phi_pic1[[11, 0, 1]].mean(axis=0),
        name="diagnostic/Hadley/total/1XPIC_Hadley_SF_DJF",
    )

    # JJA
    # %%
    p_plot(
        phi_pic2[[5, 6, 7]].mean(axis=0),
        overlay=phi_pic2[[5, 6, 7]].mean(axis=0),
        name="diagnostic/Hadley/total/2XPIC_Hadley_SF_JJA",
    )
    p_plot(
        phi_pic2adj[[5, 6, 7]].mean(axis=0),
        overlay=phi_pic2adj[[5, 6, 7]].mean(axis=0),
        name="diagnostic/Hadley/total/2XPICadj_Hadley_SF_JJA",
    )

    p_plot(
        phi_pic1[[5, 6, 7]].mean(axis=0),
        overlay=phi_pic1[[5, 6, 7]].mean(axis=0),
        name="diagnostic/Hadley/total/1XPIC_Hadley_SF_JJA",
    )

    # Difference plots

    # %%

    diff = phi_pic2adj[[11, 0, 1]].mean(axis=0) - phi_pic2[[11, 0, 1]].mean(axis=0)
    p_plot(
        diff,
        True,
        overlay=diff,
        name="diagnostic/Hadley/diff/2XPICadj_2XPIC_Hadley_SF_DJF",
        showquiv=False,
    )

    diff = phi_pic1[[11, 0, 1]].mean(axis=0) - phi_pic2[[11, 0, 1]].mean(axis=0)
    p_plot(
        diff,
        True,
        overlay=diff,
        name="diagnostic/Hadley/diff/1XPIC_2XPIC_Hadley_SF_DJF",
        showcb=False,
        showquiv=False,
    )

    # %%

    diff = phi_pic2adj[[5, 6, 7]].mean(axis=0) - phi_pic2[[5, 6, 7]].mean(axis=0)
    p_plot(
        diff,
        True,
        overlay=diff,
        name="diagnostic/Hadley/diff/2XPICadj_2XPIC_Hadley_SF_JJA",
        showquiv=False,
    )

    diff = phi_pic1[[5, 6, 7]].mean(axis=0) - phi_pic2[[5, 6, 7]].mean(axis=0)
    p_plot(
        diff,
        True,
        overlay=diff,
        name="diagnostic/Hadley/diff/1XPIC_2XPIC_Hadley_SF_JJA",
        showquiv=False,
        showcb=False,
    )
