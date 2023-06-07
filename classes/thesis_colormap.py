import cmocean
import numpy as np
from colormap import Colormap
from colormap.colors import hex2rgb
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


class ThesisPlotCols(object):
    @staticmethod
    def ITCZ() -> np.ndarray:
        cmap = cmocean.cm.matter

        my_cmap = cmap(np.arange(cmap.N))

        # Set alpha
        my_cmap[:, -1] = np.linspace(0, 1, cmap.N)

        # Create new colormap
        return ListedColormap(my_cmap)

    @staticmethod
    def optimal_scale() -> list:
        return [
            "#0b84a5",
            "#f6c85f",
            "#6f4e7c",
            "#9dd89f",
            "#ca472f",
            "#ffa056",
            "#8dddd0",
        ]

    @staticmethod
    def gray_scale() -> np.ndarray:

        return ["#000000", "#222222", "#444444", "#aaaaaa"]

    @staticmethod
    def color_scale() -> np.ndarray:

        return ["black", "red", "green", "blue", "orange", "gray", "darkblue"]

    @staticmethod
    def pastelle() -> np.ndarray:

        return [
            "#312F2F",
            "#D81E5B",
            "#FCA311",
            "#00AFB5",
            "#85B57D",
            "#ee977f",
        ]


class ThesisColormap:
    def __init__(self) -> None:
        self.c = Colormap()

    def temperature(self) -> LinearSegmentedColormap:
        collist = [
            "#960008",
            "#ff8c00",
            "#fffe00",
            "#0b6000",
            "#00bdc3",
            "#001a98",
        ]
        cl = self.__get_colmap(collist[::-1])

        return self.c.cmap(cl)
        # self.c.test_colormap(cmap)

    def statistics(self) -> LinearSegmentedColormap:
        collist = [
            "#960008",
            "#ff8c00",
            "#fffe00",
            "#0a6000",
            "#00bdc3",
            "#ffffff",
        ]
        cl = self.__get_colmap(collist[::-1])

        return self.c.cmap(cl)
        # self.c.test_colormap(cmap)

    def diverging(self) -> LinearSegmentedColormap:
        collist = [
            "#960008",
            "#fffe00",
            "#ffffff",
            "#0b6000",
            "#001a98",
        ]
        cl = self.__get_colmap(collist[::-1])

        return self.c.cmap(cl)

    def diverging_4(self) -> LinearSegmentedColormap:
        cl = [
            ["#001a98", "#00bdc3"],
            ["#00bdc3", "#146001"],
            [
                "#effc7e",
                "#ff8c00",
            ],
            ["#ff8c00", "#960008"],
        ]

        return self._get_segmented_cmap("diverging_4", cl)

    def percipitation_4(self) -> None:
        cl = [
            ["#ca3b3f", "#ef8d22"],  # white to yellow
            ["#eddc20", "#ffffff"],
            ["#ffffff", "#15ace1"],
            ["#268ac8", "#081d58"],
        ]
        return self._get_segmented_cmap("percipitation_4", cl)

    def percipitation_3(self) -> None:
        cl = [
            ["#c98244", "#e2bd9f"],
            ["#b1c0a1", "#188a69"],
            ["#26a6d3", "#101e4f"],
        ]
        return self._get_segmented_cmap("percipitation_3", cl)

    def stats(self) -> None:
        cl = [
            ["#ffffff", "#9d9d9d"],  # white to yellow
            ["#7f005f", "#ee2929"],
        ]
        return self._get_segmented_cmap("stats", cl)

    def _get_segmented_cmap(
        self, name: str, list: list[list]
    ) -> LinearSegmentedColormap:
        cmap_combined: tuple = tuple()
        for i in list:
            cmap_combined = cmap_combined + (
                self.c.cmap(self.__get_colmap(i))(np.linspace(0, 1, 128)),
            )
        return LinearSegmentedColormap.from_list(
            name, np.vstack(cmap_combined)
        )

    def __get_colmap(self, list: list[str]) -> dict:

        d: dict = {"red": [], "green": [], "blue": []}
        for i in list:
            r = hex2rgb(i)

            d["red"].append(r[0] / 255)
            d["green"].append(r[1] / 255)
            d["blue"].append(r[2] / 255)
        return d
