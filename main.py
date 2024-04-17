import numpy as np
import os
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class Specimen:
    def __init__(self, df, area):
        self.percentage_elongation = df.iloc[:, 0].values[2:]
        self.force = df.iloc[:, 1].values[2:]
        self.area = area * 1e-6

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
            self.percentage_elongation[i] / 100 + 1
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


def get_analytical_stress(specimen: Specimen, math_param):
    stress_analytical = np.zeros(len(specimen.stretch))

    c1, c2, c3, c4, c5 = math_param

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

        if I4l < 1:
            I4l = 1
        if I4c < 1:
            I4c = 1

        L = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        T = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

        s = (
            2 * c1 * B
            + 4 * c2 * c3 * (I4l - 1) * np.exp(c3 * (I4l - 1) ** 2) * F * L * F.T
            + 4 * c4 * c5 * (I4c - 1) * np.exp(c5 * (I4c - 1) ** 2) * F * T * F.T
        )

        stress_analytical[i] = s[0, 0]

    return stress_analytical


iter = 0


def objective_function(math_param):
    stress_long = get_analytical_stress(specimens[0], math_param)
    stress_circ = get_analytical_stress(specimens[1], math_param)
    error = np.sqrt(np.sum((stress_circ - specimens[1].stress) ** 2)) / len(
        specimens[1].stress
    )
    error += np.sqrt(np.sum((stress_long - specimens[0].stress) ** 2)) / len(
        specimens[0].stress
    )
    global iter
    iter += 1
    print(error, iter)
    return error


initial_guess = [2.901e06, 2.039e08, 5.167e-01, 1.931e00, 5.685e-07]
# initial_guess = [1.847e01, 2.094e06, -1.732e-05, 1.730e-03, -2.339e-08]
# initial_guess =  [ 3.046e+01,  9.245e+05, -2.311e-05,  -6.247e-04, -8.629e-9]
# result = minimize(
#     objective_function,
#     initial_guess,
#     method="Nelder-Mead",
#     tol=1e-12,
#     options={"maxiter": 250, "disp": True},
# )
# print(result)

# get_analytical_stress(specimens[1], initial_guess)

plt.figure()
plt.plot(
    specimens[0].stretch,
    get_analytical_stress(specimens[0], initial_guess),
    label="Stress",
)
plt.plot(
    specimens[0].stretch, specimens[0].stress, label="Original Stress"
)  # Original plot
plt.xlabel("Stretch")
plt.ylabel("Stress")
plt.legend()
plt.show()
