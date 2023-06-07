# %%
import os
from typing import Literal

import cmocean
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


class Plotting:
    """default cmap"""

    cmap = cmocean.cm.rain
    cbar = None
    ax: Axes = None
    cb_ticks = None
    render = None
    norm = None
    levels = None
    levs = None
    vmin = None
    vmax = None

    def __init__(
        self,
        fig_x: float = 3.4741,
        fig_y: float = 3.4741,
        **kwargs: str,  # noqa
    ) -> None:

        self.fig = plt.figure(figsize=(fig_x, fig_y))
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.size": 4,
                "xtick.major.size": 2,
                "ytick.major.size": 2,
                "xtick.major.width": 0.4,
                "ytick.major.width": 0.4,
                "text.latex.preamble": "\n".join(
                    [  # plots will use this preamble
                        r"\usepackage[detect-all]{siunitx}",
                    ]
                ),
            }
        )

    def show(
        self, colorbar: bool = False, grid: bool = True, **kwargs: str  # noqa
    ) -> None:
        self._render_original()
        if colorbar:
            self.add_colorbar(**kwargs)

        if grid:
            self.add_grid()

    def _render_original(self) -> None:
        pass

    def add_grid(self) -> None:
        pass

    def set_cmap(self, cmap: "LinearSegmentedColormap") -> None:
        self.cmap = cmap

    def savefig(self, name: str) -> None:
        if name is not None:

            ldir = "/".join(name.split("/")[:-1])
            if os.path.exists("../Figures/"):

                if not os.path.exists("../Figures/" + ldir):
                    os.makedirs("../Figures/" + ldir)
                plt.savefig(
                    "../Figures/%s.png" % name,
                    bbox_inches="tight",
                    dpi=700,
                    pad_inches=0,
                )
            else:
                print("No figures folder defined")
        plt.show()

    def set_thesis_cmap(
        self,
        cmap_s: Literal[
            "itcz", "itcz_delta", "itcz_hov", "empty", "radiative", "rain"
        ],
    ) -> None:
        if cmap_s == "itcz":
            cmap = cmocean.cm.thermal
            mp = cmap(np.arange(cmap.N))
            n = 60
            mp[0:n] = np.transpose([np.linspace(1, i, n) for i in mp[n]])
            mp[0:n, -1] = 1

            mp = ListedColormap(mp)
            self.set_cmap(mp)
        elif cmap_s == "itcz_delta":
            self.set_cmap(cmocean.cm.curl)
        elif cmap_s == "itcz_hov":
            self.set_cmap(cmocean.cm.thermal)
        elif cmap_s == "empty":
            cmap = cmocean.cm.thermal
            mp = cmap(np.arange(cmap.N))
            mp[:, -1] = 0
            mp = ListedColormap(mp)
            self.set_cmap(mp)
        elif cmap_s == "radiative":
            self.set_cmap(cmocean.cm.haline)
        elif cmap_s == "rain":
            self.set_cmap(cmocean.cm.amp)

    def add_colorbar(self, format: str = None, extend: str = "max") -> None:
        padding_pt = 10
        height_pt = 5

        pad = padding_pt / self.ax.bbox.height
        height = height_pt / self.ax.bbox.height

        ins = self.ax.inset_axes([0, -pad, 1, height])

        # self.cbax = self.fig.add_subplot(2, 1, 2)
        print(self.norm)
        self.cbar = plt.colorbar(
            self.render,
            cax=ins,
            extend=extend,
            norm=self.norm,
            ticks=self.cb_ticks,
            orientation="horizontal",
            format=format,
        )

    def set_boundary_norm(
        self, boundaries: np.ndarray | list, nlevs: int = 10
    ) -> None:

        bds = np.append(
            [
                np.linspace(boundaries[i - 1], boundaries[i], nlevs)[:-1]
                for i in range(1, len(boundaries))
            ],
            [boundaries[-1]],
        ).flatten()
        self.norm = colors.BoundaryNorm(bds, ncolors=self.cmap.N, clip=True)
        self.levs = bds

    def set_lognorm(self) -> None:
        if self.vmin is None or self.vmax is None:
            raise (Exception("Need to have vmin and vmax set"))
        self.norm = colors.LogNorm(vmin=self.vmin, vmax=self.vmax)

    def add_colorbar_unit(self, unit: str, georef: bool = False) -> None:
        if self.cbar is None:
            raise Exception("First add cbar!")
        if not georef:
            self.cbar.set_label(unit, labelpad=-12, x=-0.058)
        else:
            self.cbar.set_label(unit, labelpad=-12, x=-0.080)
