import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Данные из таблицы
lambda_angstrom = np.array([5852, 6074, 6217, 6402, 6717])  # Å
V0 = np.array([0.842, 0.770, 0.734, 0.704, 0.647])  # В
sigma_V0 = np.array([0.016, 0.017, 0.022, 0.042, 0.042])  # В

# Переводим длину волны в частоту (ω = 2πc/λ)
c = 3e8  # м/с
omega = 2 * np.pi * c / (lambda_angstrom * 1e-10)  # с⁻¹
omega_15 = omega / 1e15  # в единицах 10^15 с⁻¹

print("Частоты для построения графика:")
for i in range(len(lambda_angstrom)):
    print(f"λ = {lambda_angstrom[i]} Å, ω = {omega_15[i]:.3f} × 10¹⁵ с⁻¹")


# Линейная функция для аппроксимации V₀(ω)
def linear_func(omega, h, W):
    # V₀ = (ħω - W)/e = (hω/2π - W)/e
    # Перепишем: V₀ = (h/e)ω - (W/e)
    h_bar = h / (2 * np.pi)
    return (h_bar * omega - W) / 1.602e-19  # деление на e в СИ


# Альтернативная параметризация (проще для fitting)
def linear_func_simple(omega, slope, intercept):
    return slope * omega + intercept


# Аппроксимация линейной зависимостью
p0 = [4e-15, -2]  # начальные приближения
popt, pcov = curve_fit(linear_func_simple, omega, V0, sigma=sigma_V0, p0=p0)
slope, intercept = popt
sigma_slope, sigma_intercept = np.sqrt(np.diag(pcov))

# Расчет постоянной Планка из наклона
# slope = h/(e*2π) => h = slope * e * 2π
e = 1.602e-19  # Кл
h_exp = slope * e
sigma_h = sigma_slope * e

# Работа выхода из intercept
# intercept = -W/e => W = -intercept * e
W = -intercept * e
sigma_W = sigma_intercept * e

print(f"\nРезультаты аппроксимации:")
print(f"Наклон: {slope:.3e} ± {sigma_slope:.3e} В·с")
print(f"Пересечение: {intercept:.3f} ± {sigma_intercept:.3f} В")
print(f"Постоянная Планка: {h_exp:.3e} ± {sigma_h:.3e} Дж·с")
print(f"Табличное значение: 6.626e-34 Дж·с")
print(f"Относительная погрешность: {abs(h_exp-6.626e-34)/6.626e-34*100:.1f}%")
print(f"Работа выхода: {W:.3e} ± {sigma_W:.3e} Дж")
print(f"Работа выхода: {W/1.602e-19:.2f} ± {sigma_W/1.602e-19:.2f} эВ")

# Построение графика
plt.figure(figsize=(10, 6))

# Экспериментальные точки с погрешностями
plt.errorbar(
    omega_15,
    V0,
    yerr=sigma_V0,
    fmt="o",
    markersize=6,
    capsize=4,
    capthick=2,
    label="Экспериментальные точки",
)

textstr = "\n".join(
    (
        f"$a = ({slope*1e15:.2f} \\pm {sigma_slope*1e15:.2f}) \\times 10^{{-15}}$ В·с",
        f"$b = {intercept:.3f} \\pm {sigma_intercept:.3f}$ В",
    )
)

# Аппроксимирующая прямая
omega_fit = np.linspace(min(omega_15) - 0.1, max(omega_15) + 0.1, 100)
V0_fit = linear_func_simple(omega_fit * 1e15, slope, intercept)
plt.plot(omega_fit, V0_fit, "r-", linewidth=2, label=textstr)

# Настройки графика
plt.xlabel("Частота $\\omega$, $10^{15}$ с$^{-1}$", fontsize=12)
plt.ylabel("Запирающий потенциал $V_0$, В", fontsize=12)
plt.title("Зависимость запирающего потенциала от частоты света", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

plt.tight_layout()
plt.show()
