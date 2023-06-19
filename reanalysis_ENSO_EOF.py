# %%
from datasets import DataSets
from classes.GeoRefPlot import GeoRefPlot
from classes.TimePlot import TimePlot
import pandas as pd
import numpy as np
import xarray as xr
from eofs.standard import Eof
import cmocean
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from classes.thesis_colormap import ThesisPlotCols

ds = xr.open_dataset("Data/Observations_ENSO.nc")
lon = ds["longitude"].to_numpy()
lat = ds["latitude"].to_numpy()
sst = ds["sst"].to_numpy()[:-5, 0]
tim = ds["time"].values[:-5].reshape(-1, 12)
# %%
lon_ens = (lon < -60) | (lon > 100)
lat_ens = (lat > -30) & (lat < 30)
lon_u = lon[lon_ens]
lat_u = lat[lat_ens]

sst_enso = sst[:, :, lon_ens]
sst_enso = sst_enso[:, lat_ens]

sst_enso_y_m = DataSets.as_year_month(sst_enso)
avg = sst_enso_y_m.mean(axis=0)
sst_enso_y_m = sst_enso_y_m - avg
sst_enso = sst_enso_y_m.reshape(53 * 12, 239, 799)
# %% calculate the eof
coslat = np.cos(np.deg2rad(lat_u))
wgts = np.sqrt(coslat)[..., np.newaxis]

solver = Eof(sst_enso, weights=wgts)
# %%
eof1 = solver.eofsAsCorrelation(neofs=6)
pc1 = solver.pcs(npcs=6, pcscaling=1)

np.save("Data/eof_ERA5.npy", eof1)
np.save("Data/pc_ERA5.npy", pc1)

#%%

eof1 = np.load("Data/eof_ERA5.npy")
pc1 = np.load("Data/pc_ERA5.npy")

temp_3_4, time_t = DataSets.get_enso()


gr = GeoRefPlot(eof1[0], lat_u, lon_u, 21, add_east=False)
gr.set_zlim(-1, 1)
gr.cb_ticks = np.arange(-1, 1.1, 0.5)
gr.set_cmap(cmocean.cm.delta)
gr.add_landmask(True)

gr.show()

gr.add_colorbar()
gr.cbar.set_label(r"First Empirical Orthogonal Function")
gr.ax.set_ylim(-30, 30)
gr.ax.set_xlim(-80, 120)
gr.ax.set_yticks([-30, -15, 0, 15, 30])

gr.savefig("EOF_ERA5/EOF_era5")

tp = TimePlot(fig_scale=2)
tp.add_line(-pc1[:, 0], np.linspace(1970, 2023, 12 * (2023 - 1970)), label="PC 1")

ax2 = tp.ax.twinx()
ax2.set_ylim(-3, 3)
tp.add_line(temp_3_4, time_t, label="Nino3.4", ax=ax2, color="red")
tp.ax.set_ylim(-3, 3)
tp.set_yformat("%i")
tp.ax.set_ylabel(r"Niño3.4 \SI{}{\celsius}")
ax2.set_ylabel(r"First Principal Component", color="red")

# tp.add_legend()
tp.add_horizontal(0)
tp.ax.set_xlim(1990, 2023)
tp.show()
tp.savefig("EOF_ERA5/PCA_era5")


#%%

from classes.Plotting import Plotting
from matplotlib.ticker import FuncFormatter

pl = Plotting(fig_y=3.4731 / 1)
pl.ax = plt.subplot()
pl.ax.spines[["right", "top"]].set_visible(False)
pl.ax.set_prop_cycle(color=ThesisPlotCols.optimal_scale())

t_cor = np.array(temp_3_4[time_t > 1970], dtype=float)
fit = np.poly1d(
    np.polyfit(np.array(temp_3_4[time_t > 1970], dtype=float), -pc1[:, 0], 2)
)

pl.ax.plot([-3, 3], fit([-3, 3]), linestyle="--", color="gray", lw=0.7)


pl.ax.scatter(temp_3_4[time_t > 1970], -pc1[:, 0], s=0.5, color="black")
pl.ax.set_ylim(-3, 3)
pl.ax.set_xlim(-3, 3)
pl.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: "%i" % x))
pl.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "%i" % x))
pl.ax.set_ylabel("First Principal Component")
pl.ax.set_xlabel(r"Niño3.4 \SI{}{\celsius}")
pl.ax.text(
    1, -1.7, "Correlation: %0.2f" % np.corrcoef(t_cor, -pc1[:, 0])[0, 1], {"size": 6}
)

#%%
# December 1997 is the strongest at -26, 11
gr = GeoRefPlot(sst_enso_y_m[-26, 11], lat_u, lon_u, 21, add_east=False, mode="Equator")
# gr.pixel_render = True
gr.set_cmap(cmocean.cm.balance)
gr.set_zlim(-5, 5)
gr.add_landmask(True)
gr.show()
gr.add_colorbar("%i", extend="both")
gr.ax.set_xlim(-60, 120)
gr.ax.set_ylim(-20, 20)
gr.cbar.set_label("Temperature Anomaly \SI{}{\celsius}")
gr.savefig("SST_Dec_1997")
