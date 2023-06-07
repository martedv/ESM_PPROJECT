import numpy as np
from scipy.stats import multivariate_normal, norm
from classes.dataset import DataSet
from classes.thesis_datasets import ThesisDataSets as ds


class ITCZ(object):
    itcz_range = (ds.lat < 30) & (ds.lat > -30)

    itcz_1pic_y = np.load("../Data/time_series/itcz_1PIC_yearly.npy")
    itcz_2pic_y = np.load("../Data/time_series/itcz_2PIC_yearly.npy")
    itcz_2picadj_y = np.load("../Data/time_series/itcz_2PICADJ_yearly.npy")
    itcz_pi_y = np.load("../Data/time_series/itcz_PI_yearly.npy")
    itcz_obs_y = np.load("../Data/time_series/itcz_OBSERVATIONS_yearly.npy")

    obs_lat = np.load("../Data/time_series/obs_lat.npy")
    obs_lon = np.load("../Data/time_series/obs_lon.npy")

    itcz_1pic = np.load("../Data/time_series/itcz_1PIC.npy")
    itcz_2pic = np.load("../Data/time_series/itcz_2PIC.npy")
    itcz_2picadj = np.load("../Data/time_series/itcz_2PICADJ.npy")
    itcz_pi = np.load("../Data/time_series/itcz_PI.npy")

    @staticmethod
    def ITCZStatistics(
        data: "DataSet", months: list = [i for i in range(12)]
    ) -> np.ndarray:

        itcz_range = (ds.lat < 30) & (ds.lat > -30)
        if not data.isClimat:
            prect = data.get_variable("PRECT")
            OLR = data.get_variable("FLUT")
        else:
            # load the precipitation and the Outgoing long wave radiation
            prect = data.get_variable("PRECT")[months].mean(axis=0)
            OLR = data.get_variable("FLUT")[months].mean(axis=0)

        prect = prect[itcz_range, :]
        OLR = OLR[itcz_range, :]

        joint_cdf = np.zeros(OLR.shape)

        print(joint_cdf.shape)

        for i in range(len(ds.lon)):
            wins = [(i + j) % len(ds.lon) for j in range(7)]

            # Get the relevant percipitation and olr
            x, y = prect[:, wins], -OLR[:, wins]

            x, y = x.mean(axis=1), y.mean(axis=1)
            # we should take averages

            pos = np.dstack((x, y))

            # covariance matrix
            C = np.cov([x.flatten(), y.flatten()])

            rv = multivariate_normal([x.mean(), y.mean()], cov=C, allow_singular=True)

            jCD = rv.cdf(pos)

            joint_cdf[:, wins[3]] = jCD

        return joint_cdf

    @staticmethod
    def ITCZStatistics_NODS(
        OLR: np.ndarray, prect: np.ndarray, lon: np.ndarray
    ) -> np.ndarray:

        joint_cdf = np.zeros(OLR.shape)

        for i in range(len(lon)):
            wins = [(i + j) % len(lon) for j in range(7)]

            # Get the relevant percipitation and olr
            x, y = prect[:, wins], -OLR[:, wins]

            x, y = x.mean(axis=1), y.mean(axis=1)
            # we should take averages

            pos = np.dstack((x, y))

            # covariance matrix
            C = np.cov([x.flatten(), y.flatten()])

            rv = multivariate_normal([x.mean(), y.mean()], cov=C, allow_singular=True)

            jCD = rv.cdf(pos)

            joint_cdf[:, wins[3]] = jCD

        return joint_cdf

    @staticmethod
    def ITCZ_pdf(itcz: np.ndarray, a: float = 0.80) -> np.ndarray:

        pdff = np.zeros(itcz[0].shape)
        for i in range(len(ds.lon)):
            wins = [(i + j) % len(ds.lon) for j in range(7)]

            area = itcz[:, :, wins]

            to_s = area < a
            pdf = norm.pdf(to_s).mean(axis=(0, 2))

            pdf = pdf - np.min(pdf)
            pdff[:, i] = pdf
        return pdff

    @staticmethod
    def shift_observations(itcz: np.ndarray) -> np.ndarray:
        shifted = np.zeros(itcz.shape)

        for i in range(ds.lon.shape[0]):
            tod = itcz[:, :, [i, (i - 1) % ds.lon.shape[0]]]
            shifted[:, :, i] = tod.mean(axis=2)
        return shifted
