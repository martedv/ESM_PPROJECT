from typing import Literal

import cartopy.crs as ccrs
import cmocean
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from classes.Plotting import Plotting


class GeoRefPlot(Plotting):
    cb_ticks = None
    extent = [-180, 180, -90, 90]
    pixel_render = False
    extend = None

    def __init__(
        self,
        z: np.ndarray | np.ma.MaskedArray,
        lat: np.ndarray,
        lon: np.ndarray,
        levels: int | np.ndarray,
        add_east: bool = True,
        mode: Literal["Equator", "Full"] = "Full",
        extend: str = "both",
    ) -> None:
        fig_y = 3.4741 / 1.1

        if mode == "Equator":

            fig_y = fig_y * 0.75

        Plotting.__init__(self, fig_y=fig_y)
        self.mode = mode
        self.lat, self.lon = lat, lon

        self.z = z
        self.levels = levels
        self.extend = extend

        self.vmin = self.z.min()
        self.vmax = self.z.max()
        if add_east:
            self.lon = np.append(self.lon, [360])
            self._add_east_layer()

        self.ax = self.fig.add_subplot(
            2,
            1,
            1,
            projection=ccrs.PlateCarree(central_longitude=180),
        )

        if self.mode == "Equator":
            self.set_extent([-180, 180, -31, 31])
        self.ax.set_aspect("auto")

    def _add_east_layer(self) -> None:
        # matplotlib needs duplicate eastern layers

        if isinstance(self.z, np.ma.MaskedArray):
            mask = np.append(
                self.z.mask, np.array([self.z.mask[:, -1]]).transpose(), axis=1
            )

            self.z = np.append(self.z, np.array([self.z[:, -1]]).transpose(), axis=1)

            self.z = np.ma.masked_array(self.z, mask=mask)
        else:
            self.z = np.append(self.z, np.array([self.z[:, -1]]).transpose(), axis=1)
        pass

    def _render_original(self) -> None:
        levels = (
            np.linspace(self.vmin, self.vmax, self.levels)
            if self.levs is None
            else self.levs
        )

        if self.pixel_render:
            self.render = self.ax.pcolormesh(
                self.lon,
                self.lat,
                self.z,
                transform=ccrs.PlateCarree(),
                vmin=self.vmin,
                vmax=self.vmax,
                cmap=self.cmap,
                norm=self.norm,
            )

        else:
            self.render = self.ax.contourf(
                self.lon,
                self.lat,
                self.z,
                levels=levels,
                transform=ccrs.PlateCarree(),
                cmap=self.cmap,
                extend=self.extend,
                norm=self.norm,
            )

    def add_scatter(self, xy: np.ndarray) -> None:

        self.ax.scatter(xy[:, 0], xy[:, 1], s=0.2)

    def add_bathymetry(self) -> None:
        lm = xr.open_dataset("../Data/TopoBathyc30.nc").load()

        self.ax.pcolormesh(
            lm["longitude"][:],
            lm["latitude"][:],
            lm["Z"][:],
            cmap=cmocean.tools.lighten(cmocean.cm.topo, 0.6),
            transform=ccrs.PlateCarree(),
        )

    def add_landmask(self, present_day: bool = False) -> None:
        if present_day:

            self.ax.coastlines()
        else:
            lm = xr.open_dataset("../Data/TopoBathyc30.nc").load()
            lons = np.append(lm["longitude"][:], [180])

            Z = np.append(lm["Z"][:], np.array([lm["Z"][:][:, -1]]).transpose(), axis=1)

            self.ax.contour(
                lons,
                lm["latitude"][:],
                Z > -10,
                [0.5],
                colors=["black"],
                linewidths=[0.8],
                transform=ccrs.PlateCarree(),
            )

    def set_extent(self, extent: np.ndarray = [-180, 180, -30, 30]) -> None:
        self.ax.set_extent(extent, crs=ccrs.PlateCarree())
        self.extent = extent

    def add_grid(self) -> None:
        x_ticks = [0, 60, 120, 180, 240, 300, 359.7]

        y_ticks = [-90, -45, 0, 45, 90]

        if self.mode == "Equator":
            y_ticks = [-30, -15, 0, 15, 30]

        self.ax.xaxis.tick_top()
        self.ax.set_xticks(x_ticks, crs=ccrs.PlateCarree())
        self.ax.set_yticks(y_ticks, crs=ccrs.PlateCarree())

        self.ax.gridlines(
            linewidth=0.3,
            xlocs=np.array(x_ticks) - 180,
            ylocs=y_ticks,
            alpha=0.6,
            draw_labels=False,
        )

        lon_formatter = LongitudeFormatter(
            zero_direction_label=False,
            direction_label=True,
            transform_precision=1,
        )
        lat_formatter = LatitudeFormatter()
        self.ax.xaxis.set_major_formatter(lon_formatter)
        self.ax.yaxis.set_major_formatter(lat_formatter)
        plt.tick_params(pad=0.1)

    def set_zlim(self, vmin: float, vmax: float) -> None:
        self.vmin, self.vmax = vmin, vmax

    def add_landbox(self, extend: np.ndarray, **kwargs: str) -> None:  # noqa

        # extend is x,
        lat1, lat2 = extend[0], extend[1]
        lon1, lon2 = extend[2], extend[3]

        self.ax.add_patch(
            patches.Rectangle(
                (lat1, lon1),
                lat2 - lat1,
                lon2 - lon1,
                fill=False,
                linewidth=1,
                zorder=10,
                transform=ccrs.PlateCarree(),
                **kwargs,
            )
        )

    def add_scatter_overlay(self, variable: np.ndarray) -> None:
        xs = []
        ys = []

        for i, la_i in enumerate(variable):
            for j, it in enumerate(la_i):
                if it:
                    ys.append(self.lat[i])
                    xs.append(self.lon[j])

        self.ax.scatter(
            xs,
            ys,
            s=0.1,
            facecolor="white",
            marker="x",
            zorder=10,
            transform=ccrs.PlateCarree(),
        )
