import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Данные
lambda_values = np.array([6563, 4860, 4338, 4103])  # в Ангстремах
m_values = np.array([3, 4, 5, 6])
sigma_lambda = 20  # погрешность λ

# Вычисляем λ^-1
lambda_inv = 1 / lambda_values

# Вычисляем x = 1/2 - 1/m^2
x = 1 / 4 - 1 / m_values**2

# Погрешность для λ^-1 через метод дифференцирования
sigma_lambda_inv = sigma_lambda / lambda_values**2


# Определяем линейную функцию
def linear_func(x, k, b):
    return k * x + b


# Аппроксимация с учётом погрешностей
popt, pcov = curve_fit(
    linear_func, x, lambda_inv, sigma=sigma_lambda_inv, absolute_sigma=True
)

k, b = popt
sigma_k, sigma_b = np.sqrt(np.diag(pcov))  # стандартные отклонения коэффициентов


# Вывод коэффициентов и их погрешностей
print(f"k = {k:.6f} ± {sigma_k:.6f} [1/Å]")
print(f"b = {b:.6f} ± {sigma_b:.6f} [1/Å]")

# Построение графика
plt.errorbar(
    x,
    lambda_inv,
    yerr=sigma_lambda_inv,
    fmt="o",
    capsize=5,
    label=r"Экспериментальные данные",
)
plt.plot(x, linear_func(x, k, b), "r-", label=f"Линейная аппроксимация")
plt.xlabel(r"$1/4 - 1/m^2$")
plt.ylabel(r"$\lambda^{-1}$ [1/Å]")
plt.title(r"График $\lambda^{-1}$ от $(1/4 - 1/m^2)$ с линейной аппроксимацией")
plt.grid(True)
plt.legend()
plt.show()
