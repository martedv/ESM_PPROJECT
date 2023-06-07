import numpy as np
from classes.seasonal_energy_balance import SeasonalEnergyBalance


class TemperatureDecomp:
    eb1: "SeasonalEnergyBalance"
    eb2: "SeasonalEnergyBalance"

    t_decomp: np.ndarray

    def __init__(
        self, eb1: "SeasonalEnergyBalance", eb2: "SeasonalEnergyBalance"
    ) -> None:

        if eb1.nDims != eb2.nDims:
            raise ValueError("Dimensions of the provided EBMs do not match")

        self.eb1 = eb1
        self.eb2 = eb2

        self.calculate_decomp()

    def calculate_decomp(self) -> dict:
        HI = np.array([self.eb1.HI, self.eb2.HI])
        EM = np.array([self.eb1.EM, self.eb2.EM])
        AP = np.array([self.eb1.AP, self.eb2.AP])
        FSDTOA = self.eb1.FSDTOA

        t_decomp = dict()

        t_decomp["Total"] = self.eb1.Te - self.eb2.Te

        t_decomp["Albedo"] = self.eb1.calc_Te(
            AP[0], HI[0], EM[0], FSDTOA
        ) - self.eb1.calc_Te(AP[1], HI[0], EM[0], FSDTOA)

        t_decomp["Emissivity"] = self.eb1.calc_Te(
            AP[1], HI[0], EM[0], FSDTOA
        ) - self.eb1.calc_Te(AP[1], HI[0], EM[1], FSDTOA)

        t_decomp["Heat"] = self.eb1.calc_Te(
            AP[1], HI[0], EM[1], FSDTOA
        ) - self.eb1.calc_Te(AP[1], HI[1], EM[1], FSDTOA)
        self.t_decomp = t_decomp
        return t_decomp

    def print_0d(self) -> None:

        print("Total Temperature Difference: %0.3f K" % np.mean(self.t_decomp["Total"]))

        print("Td due to Albedo: %0.3f K" % np.mean(self.t_decomp["Albedo"]))
        print("Td due to Heat: %0.3f K" % np.mean(self.t_decomp["Heat"]))
        print("Td due to Emissivity: %0.3f K" % np.mean(self.t_decomp["Emissivity"]))
