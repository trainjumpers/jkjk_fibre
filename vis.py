import numpy as np
import os
import pandas as pd
from scipy.optimize import minimize


class Specimen:
    def __init__(self, df, area, original_length=16.5):
        self.percentage_elongation = df.iloc[:, 0].values[2:]
        self.force = df.iloc[:, 1].values[2:]
        self.area = area
        self.original_length = original_length

        assert len(self.percentage_elongation) == len(
            self.force
        ), "Length of percentage elongation and force should be the same"
        assert (
            len(self.percentage_elongation) > 4
        ), "Length of percentage elongation should be greater than 4"

        self.stretch = self._get_stretch()
        self.stress = self._get_stress()

    def __repr__(self) -> str:
        return f"Specimen(percentage_elongation={self.percentage_elongation[:3]}, force={self.force[:3]}, area={self.area}), stretch={self.stretch[:3]}"

    def _get_stretch(self):
        return [
            (self.percentage_elongation[i] * self.original_length) / 100
            for i in range(len(self.percentage_elongation))
        ]

    def _get_stress(self):
        return [self.force[i] / self.area for i in range(len(self.force))]


specimens = []


def read_xlsx_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xlsx"):
                print(f"Reading file: {os.path.join(root, file)}")
                sheet_3 = pd.read_excel(os.path.join(root, file), sheet_name=3)
                sheet_2 = pd.read_excel(os.path.join(root, file), sheet_name=2)
                area = sheet_2.iloc[1, 15]
                specimen = Specimen(sheet_3, area)
                specimens.append(specimen)


read_xlsx_files(".")

assert len(specimens) == 2, "Only 2 specimens should be found"


# Mat param are needed by scipy??
def get_analytical_stress(specimen: Specimen, mat_param):
    stress_analytical = np.zeros(len(specimen.stretch))

    c1, c2, c3, c4, c5 = mat_param
    for i in range(len(specimen.stretch)):
        if specimen.stretch[i] == 0:
            stress_analytical[i] = 0
            continue

        F = np.array(
            [
                [specimen.stretch[i], 0, 0],
                [0, 1 / np.sqrt(specimen.stretch[i]), 0],
                [0, 0, 1 / np.sqrt(specimen.stretch[i])],
            ]
        )

        B = F * F.T
        I4l = specimen.stretch[i] ** 2
        I4c = 1 / specimen.stretch[i]

        L = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        T = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

        s = (
            2 * c1 * B
            + 4 * c2 * c3 * (I4l - 1) * np.exp(c3 * (I4l - 1) ** 2) * F * L * F.T
            + 4 * c4 * c5 * (I4c - 1) * np.exp(c5 * (I4c - 1) ** 2) * F * T * F.T
        )

        stress_analytical[i] = s[0, 0]

    return stress_analytical


def objective_function(mat_param):
    stress = get_analytical_stress(specimens[0], mat_param)
    breakpoint()
    error = np.sum((stress - specimens[0].stress) ** 2)
    return error


0
initial_guess = [ 3.058e+01,  7.393e+02, -2.856e-02,  5.122e-08,  1.217e-10]
result = minimize(objective_function, initial_guess, method="Nelder-Mead", tol=1e-12)
print(result)
