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
from datetime import datetime


hist = DataSets.HIST
lat = hist["lat"]
lon = hist["lon"]
hist_ts = hist["ts"]
# mask out the land
dat = xr.open_dataset("Data/test_lsm.nc")["lsm"]
land_mask = dat[0] == 1

land_mask[(lat < -30) | (lat > 30), :] = True
land_mask[:, (lon < 100) | (lon > 300)] = True


sst = hist["ts"].to_numpy()
sst[:, land_mask] = np.nan

hist["sst"] = (("time", "lat", "lon"), sst)

hist["time"] = np.array(
    [
        datetime.strptime(date, "%Y-%m-%d %X").isoformat()
        for date in np.array(hist["time"].to_numpy(), str)
    ]
)
hist["sim_time"] = ("time", np.linspace(0, len(hist["time"]) / 12, len(hist["time"])))

del sst

hist["anom_ts"] = (
    ("time", "lat", "lon"),
    DataSets.as_monthly(
        DataSets.detrend_yearly(
            DataSets.as_year_month(hist["sst"].to_numpy()), bps=list(range(15, 151, 30))
        )
    ),
)

hist["nino3"] = (
    "time",
    hist["anom_ts"]
    .sel({"lat": slice(5, -5), "lon": slice(210, 270)})
    .mean(dim=["lat", "lon"])
    .to_numpy(),
)

hist["nino4"] = (
    "time",
    hist["anom_ts"]
    .sel({"lat": slice(5, -5), "lon": slice(160, 210)})
    .mean(dim=["lat", "lon"])
    .to_numpy(),
)

hist["nino34"] = (
    "time",
    hist["anom_ts"]
    .sel({"lat": slice(5, -5), "lon": slice(190, 240)})
    .mean(dim=["lat", "lon"])
    .to_numpy(),
)


hist.to_netcdf("Data/hist_edited.nc")
