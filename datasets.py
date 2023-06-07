import xarray as xr
import numpy as np
import pandas as pd


class DataSets:
    folder = "Data/"
    HIST = xr.open_dataset(folder + "historical_monthly_1_155.nc")

    PI = xr.open_dataset(folder + "output_allvar_monthly_1_50.nc")

    def yearly(arr: np.array):

        return DataSets.as_year_month(arr).mean(axis=1)

    def as_year_month(arr: np.ndarray):
        return arr.reshape(-1, 12, arr.shape[-2], arr.shape[-1])

    def get_enso():
        dat = pd.read_csv("Data/nino34.long.anom.data", sep="   ", header=None)

        dat.index = dat[0]

        dat = dat[:-1]

        nump_dat = dat.to_numpy()

        dates = nump_dat[:, 0]
        dates = np.linspace(np.min(dates), np.max(dates) + 1, dates.shape[0] * 12)

        data = nump_dat[:, 1:]

        data = np.reshape(nump_dat[:, 1:], np.prod(nump_dat[:, 1:].shape))

        return (data, dates)
