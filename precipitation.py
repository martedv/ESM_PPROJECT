# %%
from datasets import DataSets
from classes.GeoRefPlot import GeoRefPlot
from classes.TimePlot import TimePlot
from classes.LonPlot import LonPlot
import pandas as pd
import numpy as np
import xarray as xr
import cmocean
import cartopy.crs as ccrs
from eofs.standard import Eof
import pandas as pd
import matplotlib.pyplot as plt

hist_old = DataSets.HIST

hist = DataSets.HIST_ED

full_prc = hist_old["prc"].to_numpy()

hist_old.close()

del hist_old


lat = hist["lat"]
lon = hist["lon"]
hist["time"] = pd.date_range("1850-01-01", periods=hist.time.shape[0], freq="MS")


pc1_h = np.load("Data/pc_HIST.npy")
dat = xr.open_dataset("Data/test_lsm.nc")["lsm"]
land_mask = dat[0] < 0

land_mask[(lat < -30) | (lat > 30), :] = True
land_mask[:, (lon < 100) | (lon > 300)] = True


# hist["prc_nino_region"] = (
#     ("time", "lat", "lon"),
#     hist["prc"] * 30 * 24 * 60 * 60 * 1000,
# )
hist["nino_prc"] = (
    ("time", "lat", "lon"),
    DataSets.as_monthly(
        DataSets.detrend_yearly(
            DataSets.as_year_month(full_prc * 30 * 24 * 60 * 60 * 1000),
            bps=list(range(15, 151, 30)),
            add_lm=False,
        )
    ),
)


dat.close()
del dat
del land_mask
# %%
weights = np.cos(np.deg2rad(lat))
lon_mean_prc = hist["nino_prc"]

del weights

# el nino years as the top 10% of years with el nino
# find the years above 2 sigma = 25%

sig_pc = pc1_h[:, 0].std()
i_ninos = np.where(pc1_h[:, 0] > 1 * sig_pc)
i_ninas = np.where(pc1_h[:, 0] < -1 * sig_pc)

prc_ninos = lon_mean_prc[i_ninos].groupby(lon_mean_prc[i_ninos].time.dt.season).mean()

prc_ninas = lon_mean_prc[i_ninas].groupby(lon_mean_prc[i_ninas].time.dt.season).mean()


gr = GeoRefPlot(prc_ninos.mean("season") / 30, lat, lon, 21, add_east=False)
gr.set_cmap(cmocean.cm.delta)
gr.add_landmask(True)
gr.set_zlim(-0.8, 0.8)

gr.show()
gr.ax.set_ylim(-30, 30)
gr.ax.set_xlim(-80, 120)
gr.ax.set_yticks([-30, -15, 0, 15, 30])

gr.add_colorbar()
gr.cbar.set_label(r"Precipitation Anomaly \SI{}{\milli\metre\per\day}")
gr.cbar.set_ticks(np.arange(-0.8, 0.81, 0.2))
gr.savefig("prect/precip_el_nino_1sigma")

gr = GeoRefPlot(prc_ninas.mean("season") / 30, lat, lon, 21, add_east=False)
gr.set_cmap(cmocean.cm.delta)
gr.add_landmask(True)
gr.set_zlim(-0.8, 0.8)
gr.show()
gr.ax.set_ylim(-30, 30)
gr.ax.set_xlim(-80, 120)
gr.ax.set_yticks([-30, -15, 0, 15, 30])
gr.add_colorbar()
gr.cbar.set_label(r"Precipitation Anomaly \SI{}{\milli\metre\per\day}")
gr.cbar.set_ticks(np.arange(-0.8, 0.81, 0.2))
gr.savefig("prect/precip_el_nino_1sigma")


#%%
gr = GeoRefPlot(prc_ninos[0] / 30, lat, lon, 21, add_east=False)
gr.set_cmap(cmocean.cm.delta)
gr.add_landmask(True)
gr.set_zlim(-0.8, 0.8)

gr.show()
gr.ax.set_ylim(-30, 30)
gr.ax.set_xlim(-80, 120)
gr.ax.set_yticks([-30, -15, 0, 15, 30])

gr.add_colorbar()
gr.cbar.set_label(r"Precipitation Anomaly \SI{}{\milli\metre\per\day}")
gr.cbar.set_ticks(np.arange(-0.8, 0.81, 0.2))
gr.savefig("prect/precip_el_nino_1sigma_DJF")

gr = GeoRefPlot(prc_ninas[0] / 30, lat, lon, 21, add_east=False)
gr.set_cmap(cmocean.cm.delta)
gr.add_landmask(True)
gr.set_zlim(-0.8, 0.8)
gr.show()
gr.ax.set_ylim(-30, 30)
gr.ax.set_xlim(-80, 120)
gr.ax.set_yticks([-30, -15, 0, 15, 30])
gr.add_colorbar()
gr.cbar.set_label(r"Precipitation Anomaly \SI{}{\milli\metre\per\day}")
gr.cbar.set_ticks(np.arange(-0.8, 0.81, 0.2))
gr.savefig("prect/precip_la_nina_1sigma_DJF")

# %%
gr = GeoRefPlot(prc_ninos[1] / 30, lat, lon, 21, add_east=False)
gr.set_cmap(cmocean.cm.delta)
gr.add_landmask(True)
gr.set_zlim(-0.8, 0.8)

gr.show()
gr.ax.set_ylim(-30, 30)
gr.ax.set_xlim(-80, 120)
gr.ax.set_yticks([-30, -15, 0, 15, 30])

gr.add_colorbar()
gr.cbar.set_label(r"Precipitation Anomaly \SI{}{\milli\metre\per\day}")
gr.cbar.set_ticks(np.arange(-0.8, 0.81, 0.2))
gr.savefig("prect/precip_el_nino_1sigma_JJA")

gr = GeoRefPlot(prc_ninas[1] / 30, lat, lon, 21, add_east=False)
gr.set_cmap(cmocean.cm.delta)
gr.add_landmask(True)
gr.set_zlim(-0.8, 0.8)
gr.show()
gr.ax.set_ylim(-30, 30)
gr.ax.set_xlim(-80, 120)
gr.ax.set_yticks([-30, -15, 0, 15, 30])
gr.add_colorbar()
gr.cbar.set_ticks(np.arange(-0.8, 0.81, 0.2))
gr.cbar.set_label(r"Precipitation Anomaly \SI{}{\milli\metre\per\day}")
gr.savefig("prect/precip_la_nina_1sigma_JJA")

# %%


from classes.Plotting import Plotting
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from classes.thesis_colormap import ThesisPlotCols

pl = Plotting(fig_y=3.4731 / 2)
pl.ax = plt.subplot()
pl.ax.spines[["right", "top"]].set_visible(False)
pl.ax.set_prop_cycle(color=ThesisPlotCols.optimal_scale())

pl.ax.hist(pc1_h[:, 0], bins=np.linspace(-3, 3, 30))
pl.ax.hist(pc1_h[:, 0][i_ninos], bins=np.linspace(-3, 3, 30), label="El Niño")
pl.ax.hist(pc1_h[:, 0][i_ninas], bins=np.linspace(-3, 3, 30), label="La Niña")
pl.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: "%i" % x))
pl.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "%i" % x))
pl.ax.axvline(
    1 * sig_pc,
    0,
    200,
    color="gray",
    linestyle="--",
    alpha=0.7,
    zorder=3,
    linewidth=1,
)
pl.ax.axvline(
    -1 * sig_pc,
    0,
    200,
    color="gray",
    linestyle="--",
    alpha=0.7,
    zorder=3,
    linewidth=1,
)
pl.ax.set_xlabel(r"First Principal Component")
pl.ax.set_ylabel(r"Number of Months Occurances")
pl.ax.legend(frameon=False)
pl.savefig("nino_nina_selection")
