# %%
from datasets import DataSets
from classes.GeoRefPlot import GeoRefPlot
from classes.TimePlot import TimePlot
import pandas as pd
import numpy as np

hist = DataSets.HIST
pi = DataSets.PI


lat = hist["lat"]
lon = hist["lon"]

yearly = 1860 / 12

pi

hist_y_ts = DataSets.yearly(hist["ts"].to_numpy())
hist_ts = hist["ts"].to_numpy()


nin_lat = (lat > -5) & (lat < 5)
nin_lon = (lon > 300) & (lon < 350)


hist_y_m = DataSets.as_year_month(hist["ts"].to_numpy())

mask = np.zeros(hist_y_m.shape)
mask[:, :, :, :] = True
# mask[nin_lat, :] = False

mask[:, :, :, nin_lon] = False
mask[:, :, ~nin_lat, :] = True
s_t = np.ma.masked_array(hist_y_m, mask=mask)

window_size = 10
kernel = np.ones(window_size) / window_size

avg = np.zeros((155, 12))


temp_3_4, time_t = DataSets.get_enso()

avg[window_size - 1 :] = np.transpose(
    [np.convolve(s_t[:, i].mean(axis=(1, 2)), kernel, mode="valid") for i in range(12)]
)
avg[: window_size - 1] = avg[window_size - 1]

nino_3_4 = s_t.mean(axis=(2, 3)).reshape((1860)) - avg.reshape((1860))


tp = TimePlot()

tp.add_line(temp_3_4, time_t)
tp.add_line(
    s_t.mean(axis=(2, 3)).reshape((1860)) - avg.reshape((1860)),
    np.linspace(1850, 2005, 1860),
)


tp.ax.set_xlim(1980, 2005)
tp.ax.set_ylim(-3, 3)
