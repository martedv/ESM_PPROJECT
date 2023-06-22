import xarray as xr
import numpy as np
import pandas as pd
from scipy import signal


class DataSets:
    folder = "Data/"
    HIST = xr.open_dataset(folder + "historical_monthly_1_155.nc")
    HIST_ED = xr.open_dataset(folder + "hist_edited.nc")
    PI_ED = xr.open_dataset(folder + "PI_edited.nc")
    PI = xr.open_dataset(folder + "output_allvar_monthly_1_50.nc")

    PULSE_double = xr.open_dataset(folder + "output_standard_monthly_1_24.nc")
    PULSE_halve = xr.open_dataset(
        folder + "sim5a_pulse_double_output_standard_monthly_1_25.nc"
    )

    def nino34(arr, lat, lon):
        pass

    def yearly(arr: np.array):

        return DataSets.as_year_month(arr).mean(axis=1)

    def as_year_month(arr: np.ndarray):
        return arr.reshape(-1, 12, arr.shape[-2], arr.shape[-1])

    def as_monthly(arr: np.ndarray) -> np.ndarray:
        return arr.reshape(arr.shape[0] * 12, arr.shape[-2], arr.shape[-1])

    def get_enso():
        dat = pd.read_csv("Data/nino34.long.anom.data", sep="   ", header=None)

        dat.index = dat[0]

        dat = dat[:-1]

        nump_dat = dat.to_numpy()

        dates = nump_dat[:, 0]
        dates = np.linspace(np.min(dates), np.max(dates) + 1, dates.shape[0] * 12)

        data = nump_dat[:, 1:]

        data = np.reshape(nump_dat[:, 1:], np.prod(nump_dat[:, 1:].shape))

        return (data, np.array(dates, dtype=float))

    def convolve_yearly(arr: np.ndarray, window_size=10) -> np.ndarray:
        window_size = 10
        kernel = np.ones(window_size) / window_size

        avg = np.zeros(arr.shape)

        avg[window_size - 1 :] = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=arr
        )
        avg[: window_size - 1] = avg[window_size - 1]

        return arr - avg

    def detrend_yearly(arr: np.ndarray, bps=None, add_lm=True) -> np.ndarray:
        # find the trend for each cell for 30 years before
        # arr has shape year, months, lon, lat
        ym = arr.copy()
        ym[np.isnan(ym)] = 0

        tm = np.arange(1850, 2005.1, 1)

        detrend = signal.detrend(ym, axis=0, bp=bps)

        dat = xr.open_dataset("Data/test_lsm.nc")["lsm"]
        land_mask = dat[0] == 1
        if add_lm:
            detrend[:, :, land_mask] = np.nan

        return detrend
