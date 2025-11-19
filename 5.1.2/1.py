import matplotlib.pyplot as plt
import numpy as np

# Данные
theta = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
sigma_theta = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

inv_N = np.array(
    [
        1.199,
        1.089,
        1.222,
        1.307,
        1.385,
        1.493,
        1.845,
        2.096,
        2.242,
        2.525,
        2.770,
        3.049,
        3.279,
    ]
)
sigma_inv_N = np.array(
    [
        0.066,
        0.052,
        0.097,
        0.095,
        0.132,
        0.167,
        0.129,
        0.147,
        0.166,
        0.204,
        0.200,
        0.223,
        0.226,
    ]
)

inv_N *= 0.001
sigma_inv_N *= 0.001

# Построение графика
plt.figure(figsize=(12, 8))
plt.errorbar(
    theta,
    inv_N,
    xerr=sigma_theta,
    yerr=sigma_inv_N,
    fmt="o",
    color="blue",
    markersize=6,
    capsize=4,
    capthick=1,
    label="Экспериментальные точки",
)

# Настройки графика
plt.xlabel("Угол рассеяния θ, °", fontsize=12)
plt.ylabel("$\\frac{1}{N(θ)}$", fontsize=14)
plt.title("Зависимость $\\frac{1}{N(θ)}$ от угла рассеяния θ", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Добавление сетки
plt.minorticks_on()
plt.grid(True, which="major", alpha=0.5)
plt.grid(True, which="minor", alpha=0.2)

plt.tight_layout()
plt.show()
