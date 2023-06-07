# %%

import pandas as pd
import matplotlib.pyplot as plt

dat = pd.read_csv("Data/nino34.long.anom.data", sep="   ", header=None)

dat.index = dat[0]

dat = dat[:-1]

nump_dat = dat.to_numpy()

plt.plot(nump_dat[:, 1:].reshape(12 * 153))
