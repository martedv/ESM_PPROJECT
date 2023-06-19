# %%
from datasets import DataSets
from classes.GeoRefPlot import GeoRefPlot
from classes.TimePlot import TimePlot
import pandas as pd
import numpy as np
import xarray as xr
import cmocean
import cartopy.crs as ccrs


hist = DataSets.HIST
lat = hist["lat"]
lon = hist["lon"]
hist_y_ts = DataSets.yearly(hist["ts"].to_numpy())
hist_ts = hist["ts"].to_numpy()

hist_y_ts.shape

nin_lat = (lat > -5) & (lat < 5)
nin_lon = (lon > 190) & (lon < 240)


hist_y_m = DataSets.as_year_month(hist["ts"].to_numpy())

window_size = 30
kernel = np.ones(window_size) / window_size


avg2 = np.zeros(hist_y_m.shape)

avg2[window_size - 1 :] = np.apply_along_axis(
    lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=hist_y_m
)
avg2[: window_size - 1] = avg2[window_size - 1]


diff2 = hist_y_m - avg2


dat = xr.open_dataset("Data/test_lsm.nc")["lsm"]
land_mask = dat[0] == 1


dif_mask = np.ma.masked_array(diff2[-26, 11], mask=land_mask)


gr = GeoRefPlot(dif_mask, lat, lon, 50, mode="Equator")

gr.add_landmask(True)
gr.set_cmap(cmocean.cm.balance)
# gr.set_thesis_cmap("empty")
gr.set_zlim(-1, 1)
# gr.ax.set_yticks()
gr.show()


# gr.add_landbox([160, 210, -5, 5], color="black")
# t = gr.ax.text(165, -2, "Niño 4", transform=ccrs.PlateCarree())
# t.set_fontsize(9)
# gr.add_landbox([210, 270, -5, 5], color="black")
# t = gr.ax.text(245, -2, "Niño 3", transform=ccrs.PlateCarree())
# t.set_fontsize(9)
# gr.add_landbox([190, 240, -5, 5], color="red")
# t = gr.ax.text(195, 7, "Niño 3.5", transform=ccrs.PlateCarree(), color="red")
# t.set_fontsize(9)
gr.ax.set_yticks([-20, -10, 0, 10, 20])


gr.ax.set_xlim(-60, 120)
gr.ax.set_ylim(-20, 20)
gr.savefig("nino_event_1")

# %%

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


tp = TimePlot()
tp.add_line(dif_3_4_t, np.linspace(1850, 2005, 1860))
tp.add_line(temp_3_4, time_t)


tp.ax.set_xlim(1980, 2005)
tp.ax.set_ylim(-3, 3)

# mask = np.zeros(hist_y_m.shape)
# mask[:, :, :, :] = True
# # mask[nin_lat, :] = False

# mask[:, :, :, nin_lon] = False
# mask[:, :, ~nin_lat, :] = True
# s_t = np.ma.masked_array(hist_y_m, mask=mask)

# window_size = 10
# kernel = np.ones(window_size) / window_size

# avg = np.zeros((155, 12))


# nin_3_4_true, dates = DataSets.get_enso()


# avg[window_size - 1 :] = np.transpose(
#     [np.convolve(s_t[:, i].mean(axis=(1, 2)), kernel, mode="valid") for i in range(12)]
# )
# avg[: window_size - 1] = avg[window_size - 1]

# nino_3_4 = s_t.mean(axis=(2, 3)).reshape((1860)) - avg.reshape((1860))


# tp = TimePlot()

# tp.add_line(nino_3_4, np.linspace(1850, 2005, 1860))
# tp.add_line(nin_3_4_true, dates)


# tp.ax.set_xlim(1990, 2005)
# tp.ax.set_ylim(-2, 2)
