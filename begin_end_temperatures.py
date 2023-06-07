# %%
from datasets import DataSets
from classes.GeoRefPlot import GeoRefPlot

hist = DataSets.HIST
lat = hist["lat"]
lon = hist["lon"]

yearly = 1860 / 12

hist_y_ts = DataSets.yearly(hist["ts"].to_numpy())

gr = GeoRefPlot(hist_y_ts[-1] - hist_y_ts[0], lat, lon, 20)
gr.add_landmask(True)
gr.set_thesis_cmap("itcz_delta")
gr.set_zlim(-10, 10)
gr.show()
gr.add_colorbar("%i")
gr.cbar.set_label("Temperature K")
