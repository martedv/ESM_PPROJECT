# %%
import os
from datasets import DataSets
from classes.GeoRefPlot import GeoRefPlot
import numpy as np
from scipy.optimize import curve_fit


PULSE_double = DataSets.PULSE_double.mean("lon")


PULSE_double["sim_time"] = (
    "time",
    np.linspace(0, len(PULSE_double["time"]) / 12 - 1 / 12, len(PULSE_double["time"])),
)

weights = np.cos(np.deg2rad(PULSE_double.lat))
halve_weighted = PULSE_double.weighted(weights)


dmean = halve_weighted.mean("lat").groupby(np.floor(PULSE_double["sim_time"])).mean()


R = np.zeros(len(dmean["sim_time"]) - 1)


T = dmean.ts - dmean.ts[0]

F = dmean.rst + dmean.rlut  # net F over time


F2x = 3.74
C0 = 280
C = np.ones(len(PULSE_double["sim_time"])) * (C0 * 2)
C[0] = C0
F = np.log(C / C0) * F2x / np.log(2)


R[0] = T[1] / F[1]
for t in range(1, len(dmean["sim_time"]) - 1):
    fac = 0
    for tau in range(1, t + 1):

        fac += R[t - tau] * F[tau + 1]
    R[t] = (T[t + 1] - fac) / F[1]


#%%
def response(x, q1, q2, d1, d2):
    return (q1 / d1) * np.exp(-x / d1) + (q2 / d2) * np.exp(-x / d2)


q1_s = 0.33
q2_s = 0.41
d1_s = 239  # years
d2_s = 4.1  # years

initial_guesses = [q1_s, q2_s, d1_s, d2_s]
fit, acc = curve_fit(response, dmean.sim_time[1:-1], R[1:], p0=initial_guesses)
# now we have a guess for the fit of R
# we can
# we can access it by
#%%
import matplotlib.pyplot as plt

plt.plot(dmean.sim_time[:-1], R)
plt.plot(dmean.sim_time, response(dmean.sim_time, *fit))
plt.plot(dmean.sim_time, response(dmean.sim_time, *initial_guesses))
