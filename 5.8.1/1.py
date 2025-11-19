import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib

# Данные из таблицы
T_light = np.array(
    [1143, 1243, 1343, 1443, 1543, 1643, 1743, 1843, 1943, 2043, 2143, 2243, 2343]
)  # Яркостная температура, K
T_real = np.array(
    [1158, 1258, 1368, 1488, 1588, 1693, 1793, 1903, 2018, 2123, 2233, 2338, 2443]
)  # Термодинамическая температура, K
sigma_T = np.array(
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
)  # Погрешность температуры, K

I = np.array(
    [
        0.504,
        0.546,
        0.581,
        0.625,
        0.665,
        0.703,
        0.754,
        0.818,
        0.850,
        0.892,
        0.953,
        0.997,
        1.042,
    ]
)  # Ток, A
sigma_I = np.array(
    [
        0.003,
        0.004,
        0.004,
        0.004,
        0.004,
        0.005,
        0.005,
        0.005,
        0.005,
        0.006,
        0.006,
        0.006,
        0.006,
    ]
)  # Погрешность тока, A

V = np.array(
    [
        1.582,
        1.973,
        2.309,
        2.768,
        3.192,
        3.615,
        4.220,
        5.002,
        5.417,
        5.979,
        6.825,
        7.477,
        8.162,
    ]
)  # Напряжение, V
sigma_V = np.array(
    [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]
)  # Погрешность напряжения, V

eps_T = np.array(
    [
        0.12712,
        0.13938,
        0.15760,
        0.17720,
        0.19308,
        0.20802,
        0.22202,
        0.23639,
        0.25134,
        0.26499,
        0.27929,
        0.29294,
        0.30659,
    ]
)

# Расчет мощности и ее погрешности
W = I * V / eps_T  # Мощность, Вт
sigma_W = W * np.sqrt((sigma_I / I) ** 2 + (sigma_V / V) ** 2)  # Погрешность мощности

# Логарифмирование данных
ln_T = np.log(T_real)
ln_W = np.log(W)

# Погрешности в логарифмическом масштабе
sigma_ln_T = sigma_T / T_real
sigma_ln_W = sigma_W / W

# Линейная регрессия
slope, intercept, r_value, p_value, std_err = stats.linregress(ln_T, ln_W)

# Расчет погрешностей параметров
n = len(ln_T)
S_xx = np.sum((ln_T - np.mean(ln_T)) ** 2)
sigma_slope = std_err * np.sqrt(n / S_xx)
sigma_intercept = std_err * np.sqrt(np.sum(ln_T**2) / S_xx)

# Создание графика
plt.figure(figsize=(12, 8))

# Построение точек с погрешностями
plt.errorbar(
    ln_T,
    ln_W,
    xerr=sigma_ln_T,
    yerr=sigma_ln_W,
    fmt="o",
    markersize=6,
    capsize=4,
    capthick=1.5,
    ecolor="black",
    elinewidth=1,
    markerfacecolor="blue",
    markeredgecolor="black",
    markeredgewidth=0.5,
    label="Экспериментальные точки",
)

# Построение линии регрессии
x_fit = np.linspace(min(ln_T) - 0.1, max(ln_T) + 0.1, 100)
y_fit = intercept + slope * x_fit
plt.plot(
    x_fit,
    y_fit,
    "r-",
    linewidth=2,
    label=f"Линейная аппроксимация: $y = a + n \cdot x$",
)

# Настройка внешнего вида графика
plt.xlabel("$\ln(T)$, K", fontsize=14)
plt.ylabel("$\ln(W)$, Вт", fontsize=14)
plt.title(
    "Зависимость логарифма мощности от логарифма температуры\nдля вольфрамовой нити",
    fontsize=16,
    pad=20,
)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, loc="lower right")

# Добавление информации о параметрах в левом верхнем углу
textstr = f"$n = {slope:.2f} \pm {sigma_slope:.2f}$\n$a = {intercept:.2f} \pm {sigma_intercept:.2f}$"
props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
plt.text(
    0.02,
    0.98,
    textstr,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=props,
)

# Улучшение читаемости осей
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()

# Вывод дополнительной информации
print("РЕЗУЛЬТАТЫ АППРОКСИМАЦИИ:")
print(f"Угловой коэффициент (n) = {slope:.3f} ± {sigma_slope:.3f}")
print(f"Коэффициент смещения (a) = {intercept:.3f} ± {sigma_intercept:.3f}")
print(f"Коэффициент детерминации R² = {r_value**2:.4f}")
print(f"Теоретическое значение n = 4.000")
print(f"Относительное расхождение: {abs(4-slope)/4*100:.1f}%")
