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

# %%
hist = DataSets.HIST
lat = hist["lat"].to_numpy()
lon = hist["lon"]
hist_y_ts = DataSets.yearly(hist["ts"].to_numpy())
hist_ts = hist["ts"].to_numpy()

dat = xr.open_dataset("Data/test_lsm.nc")["lsm"]
land_mask = dat[0] == 1


hist_y_ts.shape

nin_lat = (lat > -5) & (lat < 5)
nin_lon = (lon > 190) & (lon < 240)


hist_y_m = DataSets.as_year_month(hist["ts"].to_numpy())

window_size = 10
kernel = np.ones(window_size) / window_size


avg2 = np.zeros(hist_y_m.shape)

avg2[window_size - 1 :] = np.apply_along_axis(
    lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=hist_y_m
)
avg2[: window_size - 1] = avg2[window_size - 1]


diff2 = hist_y_m - avg2

coslat = np.cos(np.deg2rad(lat))
wgts = np.sqrt(coslat)[..., np.newaxis]


land_mask[(lat < -30) | (lat > 30), :] = True
land_mask[:, (lon < 100) | (lon > 300)] = True

reshaped_diff2 = diff2.reshape(155 * 12, 64, 128)

# reshaped_diff2 = np.apply_along_axis(lambda x: np.ma.masked_array(x,mask=land_mask),axis=0,arr=reshaped_diff2)
reshaped_diff2[:, land_mask] = np.nan


# %% Do the EOF


solver = Eof(reshaped_diff2, weights=wgts)
# %%
eof1 = solver.eofsAsCorrelation(neofs=7)
pc1 = solver.pcs(npcs=7, pcscaling=1)
# %%
dat = xr.open_dataset("Data/test_lsm.nc")["lsm"]
land_mask = dat[0] == 1

temp_3_4, time_t = DataSets.get_enso()


mask_3_4 = land_mask.copy()
mask_3_4[:, :] = False

mask_3_4[~nin_lat, :] = True
mask_3_4[:, ~nin_lon] = True

mask_3_4 = np.repeat(
    np.repeat(mask_3_4.to_numpy()[np.newaxis, :], 12, axis=0)[np.newaxis, :],
    diff2.shape[0],
    axis=0,
)

dif_3_4 = np.ma.masked_array(diff2, mask=mask_3_4)

dif_3_4_t = dif_3_4.mean(axis=(2, 3)).reshape(np.prod(mask_3_4.shape[0:2]))


# %%
for i in range(2):
    gr = GeoRefPlot(eof1[i], lat, lon, 21, mode="Equator")
    gr.set_cmap(cmocean.cm.delta)
    gr.set_zlim(-1, 1)
    gr.cb_ticks = np.arange(-1, 1.1, 0.5)
    gr.add_landmask(True)

    gr.show()
    gr.add_colorbar()
    gr.ax.set_ylim(-30, 30)
    gr.ax.set_xlim(-80, 120)
    gr.savefig("EOFS/EOF_%i" % i)

    tp = TimePlot()
    tp.add_line(pc1[:, i] / 3, np.linspace(1850, 2005, 1860), label="PC %i" % (i + 1))
    tp.add_line(dif_3_4_t, np.linspace(1850, 2005, 1860), label="Nino3.4")
    tp.add_legend()
    tp.ax.set_xlim(1950, 2000)
    tp.show()

    tp.savefig("EOFS/PCA_%i" % i)

# %%
np.corrcoef(pc1[:, 0], dif_3_4_t)

# Thermocline depth -5,5,120e,280e
# Zonal windstress anomaly -2.5,2.5,120e,200e
