import numpy as np
import os
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


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
        print(self.percentage_elongation)
        return [
            self.percentage_elongation[i] / 100 + 1 if self.percentage_elongation[i] > 0 else 0
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


def get_analytical_stress(specimen: Specimen, mat_param):
    stress_analytical = np.zeros(len(specimen.stretch))

    c1, c2, c3 = mat_param

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

        L = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        T = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

        P = (
            2 * c1 * F
            + 4 * c2 * c3 * (I4l - 1) * np.exp(c3 * (I4l - 1) ** 2) * F * L
        )

        p=2*c1/specimen.stretch[i]

        stress_analytical[i] = P[0, 0]-p

    return stress_analytical


iter = 0


def objective_function(mat_param):
    mat1 = [mat_param[0], mat_param[1], mat_param[2]]   
    mat2 = [mat_param[0], mat_param[3], mat_param[4]]

    stress_long = get_analytical_stress(specimens[0], mat1)
    stress_circ = get_analytical_stress(specimens[1], mat2)

    error = np.sqrt(np.sum((stress_circ - specimens[1].stress) ** 2)) / len(
        specimens[1].stress
    )
    error += np.sqrt(np.sum((stress_long - specimens[0].stress) ** 2)) / len(
        specimens[0].stress
    )
    global iter
    iter += 1
    print(f"iter: {iter} error: {error}")
    print(mat_param)
    print("=====================")
    return error



initial_guess = [2e09,  3e8, -2.5,  3.2e8, -2.5] # takes 1200 function evaluations
# initial_guess = [ 2.01808154e+09,  3.06665660e+08, -2.64567377e+00,  3.24740363e+08, -2.50634237e+00] # takes 520 function evaluations

result = minimize(
    objective_function,
    initial_guess,
    method="Nelder-Mead",
    tol=1e-6,
    options={"maxiter": 2500, "disp": True},
    bounds=[(0,None), (0,None), (None,None), (0,None), (None,None)]
)
print(result)
final_guess = result.x

plt.figure(1)
plt.plot(
    specimens[0].stretch,
    get_analytical_stress(specimens[0], final_guess[:3]),
    label="After fitting",
)
plt.plot(
    specimens[0].stretch,
    get_analytical_stress(specimens[0], initial_guess[:3]),
    label="Initial Guess",
)
plt.plot(
    specimens[0].stretch, specimens[0].stress, label="Experimental Stress"
)
plt.xlabel("Stretch")
plt.ylabel("Stress")
plt.legend()
plt.xlim(1, max(specimens[0].stretch))
plt.title("Specimen 1")

plt.figure(2)
plt.plot(
    specimens[1].stretch,
    get_analytical_stress(specimens[1], [final_guess[0],final_guess[-2], final_guess[-1]]),
    label="After fitting",
)
plt.plot(
    specimens[1].stretch,
    get_analytical_stress(specimens[1], [initial_guess[0],initial_guess[-2], initial_guess[-1]]),
    label="Initial Guess",
)
plt.plot(
    specimens[1].stretch, specimens[1].stress, label="Experimenta Stress"
)  # Original plot
plt.xlabel("Stretch")
plt.ylabel("Stress")
plt.legend()
plt.xlim(1, max(specimens[1].stretch))
plt.title("Specimen 2")

plt.figure(3)
for i in range(3):
    new_final_guess = [final_guess[0]+0.01*(i-1)*final_guess[0],final_guess[1],final_guess[2],final_guess[3],final_guess[4]]

    corr, _ = pearsonr(specimens[1].stretch, get_analytical_stress(specimens[1], [new_final_guess[0],new_final_guess[-2], new_final_guess[-1]]))
    
    plt.plot(
        specimens[1].stretch,
        get_analytical_stress(specimens[1], [new_final_guess[0],new_final_guess[-2], new_final_guess[-1]]),
        label="c1 = {:.2f}".format(new_final_guess[0]/10**8)+'e8'+', pearson_r = {:.3f}'.format(corr)
    )
 

plt.xlim(1, max(specimens[1].stretch))
plt.legend()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Stretch", fontsize=12)
plt.ylabel("Stress", fontsize=12)
plt.title("Sensitivity Analysis of c1")


plt.figure(4)
for i in range(3):
    new_final_guess = [final_guess[0],final_guess[1],final_guess[2],final_guess[3]+0.01*(i-1)*final_guess[3],final_guess[4]]

    corr, _ = pearsonr(specimens[1].stretch, get_analytical_stress(specimens[1], [new_final_guess[0],new_final_guess[-2], new_final_guess[-1]]))

    plt.plot(
        specimens[1].stretch,
        get_analytical_stress(specimens[1], [new_final_guess[0],new_final_guess[-2], new_final_guess[-1]]),
        label="c4 = {:.2f}".format(new_final_guess[3]/10**7)+'e7'+', pearson_r = {:.3f}'.format(corr)
    )
plt.xlim(1, max(specimens[1].stretch))
plt.legend()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Stretch", fontsize=12)
plt.ylabel("Stress", fontsize=12)
plt.title("Sensitivity Analysis of c4")

plt.figure(5)
for i in range(3):
    new_final_guess = [final_guess[0],final_guess[1],final_guess[2],final_guess[3],final_guess[4]+0.01*(i-1)*final_guess[4]]

    corr, _ = pearsonr(specimens[1].stretch, get_analytical_stress(specimens[1], [new_final_guess[0],new_final_guess[-2], new_final_guess[-1]]))
    
    plt.plot(
        specimens[1].stretch,
        get_analytical_stress(specimens[1], [new_final_guess[0],new_final_guess[-2], new_final_guess[-1]]),
        label="c5 = {:.2f}".format(new_final_guess[4])+ ', pearson_r = {:.3f}'.format(corr)
    )
plt.xlim(1, max(specimens[1].stretch))
plt.legend()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Stretch", fontsize=12)
plt.ylabel("Stress", fontsize=12)
plt.title("Sensitivity Analysis of c5")

plt.show()
