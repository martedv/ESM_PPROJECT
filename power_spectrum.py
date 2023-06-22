# %%
from scipy import signal
from classes.Plotting import Plotting
from sklearn.preprocessing import normalize
import numpy as np
from datasets import DataSets
import matplotlib.pyplot as plt
from classes.Plotting import Plotting
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from classes.thesis_colormap import ThesisPlotCols


hist_ed = DataSets.HIST_ED
pi_ed = DataSets.PI_ED

temp_3_4, time_t = DataSets.get_enso()
eof_PI = np.load("Data/eof_PI.npy")
pca_PI = np.load("Data/pc_PI.npy")

eof_PI = np.load("Data/eof_PI.npy")
pca_PI = np.load("Data/pc_PI.npy")


xs, fs = signal.welch(np.array(temp_3_4, float), 12, detrend=False, nperseg=12 * 25)


xs_PLASIM, fs_PLASIM = signal.welch(
    np.array(hist_ed["nino34"], float), 12, detrend=False, nperseg=12 * 25
)

xs_PLASIM_PI, fs_PLASIM_PI = signal.welch(
    np.array(pi_ed["nino34"], float), 12, detrend=False, nperseg=12 * 25
)

fs = fs / np.sum(fs)
fs_PLASIM = fs_PLASIM / np.sum(fs_PLASIM)
fs_PLASIM_PI = fs_PLASIM_PI / np.sum(fs_PLASIM_PI)


def invert(x):
    # 1/x with special treatment of x == 0
    x = np.array(x).astype(float)
    near_zero = np.isclose(x, 0)
    x[near_zero] = 1e30
    x[~near_zero] = 1 / x[~near_zero]
    return x


pl = Plotting(fig_y=3.4741 / 3)
pl.ax = plt.subplot()
years = np.append(np.arange(2, 9.1, 1), 13)

pl.ax.spines[["right", "top"]].set_visible(False)
pl.ax.set_prop_cycle(color=ThesisPlotCols.optimal_scale())

pl.ax.plot(xs, fs, label="Niño3.4 reanalysis", linewidth=0.55)
pl.ax.plot(xs_PLASIM, fs_PLASIM, label="Niño3.4 PLASIM PI", linewidth=0.55)
pl.ax.plot(xs_PLASIM_PI, fs_PLASIM_PI, label="Niño3.4 PLASIM HIST", linewidth=0.55)
pl.ax.set_xlim(0, 0.8)
pl.ax.set_xticks(np.arange(0, 0.81, 0.1))

secax = pl.ax.secondary_xaxis("top", functions=(invert, invert))
secax.set_xticks(np.append(years, 1e30), np.append(np.array(years, int), "Inf"))
pl.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "%0.1f" % x))
pl.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: "%0.2f" % x))

pl.ax.set_ylim(0, 0.18)

pl.ax.set_xlabel(r"Frequency \SI{}{1\per yr}")
secax.set_xlabel(r"Period \SI{}{yr}")
plt.legend(frameon=False)

pl.savefig("Power_spectrum")
