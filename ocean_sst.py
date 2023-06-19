# %%
import xarray as xr
from datasets import DataSets
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from classes.GeoRefPlot import GeoRefPlot
import cmocean
import numpy as np

ds = xr.open_dataset("Data/hist_ocean.nc")
pot_t = ds["var92"]

land = ds["var40"]


SST = np.ma.masked_array(pot_t[:][14, 0], mask=(land[0, 0] == 0))

hist = DataSets.HIST
lat = hist["lat"].to_numpy()
lon = hist["lon"]
hist_ts = hist["ts"].to_numpy()
gr = GeoRefPlot(hist_ts[0], lat, lon, 10)
gr.set_thesis_cmap("empty")
gr.show()
gr.set_cmap(cmocean.cm.thermal)
gr.render = gr.ax.pcolormesh(
    pot_t.xvals,
    pot_t.yvals,
    SST,
    transform=ccrs.PlateCarree(),
    zorder=1,
    cmap=gr.cmap,
    vmin=300,
)
gr.add_landmask(True)
gr.ax.set_yticks([-20, -10, 0, 10, 20])


gr.ax.set_xlim(-60, 120)
gr.ax.set_ylim(-20, 20)

gr.add_colorbar()
