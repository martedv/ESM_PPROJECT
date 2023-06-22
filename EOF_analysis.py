# %%
from datasets import DataSets
from classes.GeoRefPlot import GeoRefPlot
from classes.TimePlot import TimePlot
import pandas as pd
import numpy as np
import xarray as xr
import cmocean
import cartopy.crs as ccrs
from eofs.standard import Eof
import pandas as pd


hist_ed = DataSets.HIST_ED

# hist_ed = hist_s.sel(time=hist_s["time"][120:])
lat = hist_ed.lat
lon = hist_ed.lon

# %%
coslat = np.cos(np.deg2rad(lat.to_numpy()))
wgts = np.sqrt(coslat)[..., np.newaxis]

solver = Eof(hist_ed["anom_ts"].to_numpy(), weights=wgts)


# %%
eof1 = solver.eofsAsCorrelation(neofs=4)
pc1 = solver.pcs(npcs=4, pcscaling=1)

# np.save("Data/eof_HIST.npy",eof1)
# np.save("Data/pc_HIST.npy",pc1)

temp_3_4, time_t = DataSets.get_enso()

for i in range(2):
    gr = GeoRefPlot(-eof1[i], lat, lon, 21)
    gr.set_cmap(cmocean.cm.delta)
    gr.set_zlim(-1, 1)
    gr.cb_ticks = np.arange(-1, 1.1, 0.5)
    gr.add_landmask(True)

    gr.show()
    gr.add_colorbar()
    gr.ax.set_ylim(-30, 30)
    gr.ax.set_xlim(-80, 120)
    gr.ax.set_yticks([-30, -15, 0, 15, 30])
    gr.cbar.set_label(r"First Empirical Orthogonal Function")
    gr.savefig("EOFS/EOF_%i_PI" % i)

    tp = TimePlot(fig_scale=2)
    # tp.add_line(hist_ed["nino34"], hist_ed["sim_time"], label="Nino3.4")
    tp.add_line(hist_s["nino34"], hist_s["sim_time"], label="Nino3.4")
    ax2 = tp.ax.twinx()
    tp.add_line(
        pc1[:, i],
        hist_ed["sim_time"],
        label="PC %i" % (i + 1),
        color="red",
        ax=ax2,
    )

    ax2.set_ylim(-3, 3)

    tp.ax.set_ylim(-1, 1)
    tp.set_yformat("%0.1f")
    tp.ax.set_ylabel(r"Niño3.4 \SI{}{\celsius}")
    ax2.set_ylabel(r"First Principal Component", color="red")
    tp.add_horizontal(0)
    tp.ax.set_xlim(0, 50)
    tp.show()

    tp.savefig("EOFS/PCA_%i_PI" % i)

# %%
temp_3_4, time_t = DataSets.get_enso()

eof1_ERA5 = np.load("Data/eof_ERA5.npy")
pc1_ERA5 = np.load("Data/pc_ERA5.npy")
ERA_5_time = np.linspace(1970, 2023, 12 * (2023 - 1970))

from classes.Plotting import Plotting
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from classes.thesis_colormap import ThesisPlotCols

pl = Plotting(fig_y=3.4731 / 1)
pl.ax = plt.subplot()
pl.ax.spines[["right", "top"]].set_visible(False)
pl.ax.set_prop_cycle(color=ThesisPlotCols.optimal_scale())

# t_cor = np.array(temp_3_4[time_t > 1970], dtype=float)
fit = np.poly1d(np.polyfit(hist_ed["nino34"], pc1[:, 0], 2))
fit_ERA = np.poly1d(
    np.polyfit(np.array(temp_3_4[time_t >= 1970], float), -pc1_ERA5[:, 0], 2)
)

pl.ax.plot([-0.9, 0.9], fit([-0.9, 0.9]), linestyle="--", color="gray", lw=0.7)
pl.ax.plot([-3, 3], fit_ERA([-3, 3]), linestyle="--", color="darkred", lw=0.7)


pl.ax.scatter(hist_ed["nino34"], pc1[:, 0], s=0.5, color="black")

pl.ax.scatter(temp_3_4[time_t >= 1970], -pc1_ERA5[:, 0], s=0.5, color="red")
pl.ax.set_ylim(-3, 3)
pl.ax.set_xlim(-3, 3)
pl.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: "%i" % x))
pl.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "%i" % x))
pl.ax.set_ylabel("First Principal Component")
pl.ax.set_xlabel(r"Niño3.4 \SI{}{\celsius}")
pl.ax.text(
    1,
    -1.7,
    "Correlation: %0.2f" % np.corrcoef(hist_ed["nino34"], pc1[:, 0])[0, 1],
    {"size": 6},
)
pl.ax.text(
    1,
    -1.96,
    "Correlation: %0.2f"
    % np.corrcoef(np.array(temp_3_4[time_t >= 1970], float), -pc1_ERA5[:, 0])[0, 1],
    {"size": 6, "color": "red"},
)
pl.savefig("correlation_graph")

# %%

from scipy import signal
from classes.Plotting import Plotting
from sklearn.preprocessing import normalize

xs, fs = signal.welch(np.array(temp_3_4, float), 12, detrend=False, nperseg=12 * 25)
xs_PLASIM, fs_PLASIM = signal.welch(
    np.array(hist_ed["nino34"], float), 12, detrend=False, nperseg=12 * 25
)

xs_PLASIM_PCA, fs_PLASIM_PCA = signal.welch(
    np.array(pc1[:, 0], float), 12, detrend=False, nperseg=12 * 25
)

fs = fs / np.sum(fs)
fs_PLASIM_PCA = fs_PLASIM_PCA / np.sum(fs_PLASIM_PCA)
fs_PLASIM = fs_PLASIM / np.sum(fs_PLASIM)


def invert(x):
    # 1/x with special treatment of x == 0
    x = np.array(x).astype(float)
    near_zero = np.isclose(x, 0)
    x[near_zero] = 1e30
    x[~near_zero] = 1 / x[~near_zero]
    return x


pl = Plotting(fig_y=3.4741 / 3)
pl.ax = plt.subplot()
years = np.append(np.arange(2, 9.1, 1), 13)

pl.ax.spines[["right", "top"]].set_visible(False)
pl.ax.set_prop_cycle(color=ThesisPlotCols.optimal_scale())
pl.ax.plot(xs, fs, label="Niño3.4 reanalysis", linewidth=0.55)
pl.ax.plot(xs_PLASIM, fs_PLASIM, label="Niño3.4 PLASIM", linewidth=0.55, color="red")

# pl.ax.plot(xs_PLASIM_PCA, fs_PLASIM_PCA, label="Niño3.4 PLASIM", linewidth=0.55, color="red")
pl.ax.set_xlim(0, 0.8)
pl.ax.set_xticks(np.arange(0, 0.81, 0.1))

secax = pl.ax.secondary_xaxis("top", functions=(invert, invert))
secax.set_xticks(np.append(years, 1e30), np.append(np.array(years, int), "Inf"))
pl.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "%0.1f" % x))
pl.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: "%0.2f" % x))

pl.ax.set_ylim(0, 0.18)

pl.ax.set_xlabel(r"Frequency \SI{}{1\per yr}")
secax.set_xlabel(r"Period \SI{}{yr}")
plt.legend(frameon=False)

pl.savefig("Power_spectrum")

# Thermocline depth -5,5,120e,280e
# Zonal windstress anomaly -2.5,2.5,120e,200e
