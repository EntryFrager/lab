import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- 1. Входные данные (те же, что и раньше) ---
I = np.array(
    [
        0.00,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.2,
        1.4,
        1.6,
        1.8,
        2.0,
        2.2,
        2.4,
        2.6,
        2.8,
        3.0,
        3.2,
        3.4,
        3.6,
        3.8,
        3.85,
        3.90,
        3.95,
        4.00,
        4.05,
        4.10,
        4.15,
        4.20,
        4.25,
        4.30,
        4.35,
        4.40,
        4.45,
        4.50,
        4.55,
        4.60,
        4.65,
        4.70,
        4.75,
        4.80,
        5.00,
    ]
)

N_measured = np.array(
    [
        0.480,
        0.570,
        0.520,
        0.560,
        0.680,
        0.790,
        1.030,
        1.360,
        1.799,
        2.439,
        2.929,
        3.959,
        4.399,
        4.489,
        4.919,
        4.189,
        3.769,
        2.809,
        1.570,
        1.370,
        1.779,
        1.999,
        2.629,
        3.389,
        4.299,
        5.688,
        6.818,
        8.618,
        8.598,
        8.298,
        6.958,
        6.488,
        4.639,
        3.769,
        2.689,
        1.650,
        1.180,
        0.920,
        0.740,
        0.670,
        0.340,
    ]
)

# Константы
tau = 100.0
k = 238.0
sigma_k = 1.0
sigma_I_const = 0.02
m0c2 = 511.0

# --- 2. Подготовка физических величин ---
pc = I * k
E_total = np.sqrt(pc**2 + m0c2**2)
T = E_total - m0c2

# Погрешности N и T
sigma_N = np.sqrt(N_measured / tau)
sigma_N[sigma_N == 0] = 1.0 / tau

sigma_pc = np.sqrt((k * sigma_I_const) ** 2 + (I * sigma_k) ** 2)
dT_dpc = pc / E_total
sigma_T = np.abs(dT_dpc) * sigma_pc

# Фон
slope_bg = (N_measured[-1] - N_measured[0]) / (I[-1] - I[0])
intercept_bg = N_measured[0] - slope_bg * I[0]
N_background = slope_bg * I + intercept_bg
N_corrected = N_measured - N_background

# --- 3. Расчет координат Ферми-Кюри ---

# Фильтруем данные:
# 1. Ток I > 0.2 (чтобы импульс p не был около нуля, деление на ноль)
# 2. N_corrected > 0 (чтобы можно было извлечь корень)
mask_fermi = (I > 0.5) & (N_corrected > 0)

# Применяем маску
T_fermi = T[mask_fermi]
pc_fermi = pc[mask_fermi]
N_corr_fermi = N_corrected[mask_fermi]
sigma_N_fermi = sigma_N[mask_fermi]
sigma_pc_fermi = sigma_pc[mask_fermi]
sigma_T_fermi = sigma_T[mask_fermi]  # Горизонтальные усы

# Функция Ферми Y = sqrt(N) / p^1.5 * 1e6
# Разбиваем на части для удобства
numerator = np.sqrt(N_corr_fermi)
denominator = pc_fermi**1.5
Y_fermi = (numerator / denominator) * 1e6

# --- 4. Расчет вертикальной погрешности (sigma_Y) ---
# Y зависит от N и от p. Используем формулу переноса погрешностей:
# (dY/Y)^2 = (0.5 * dN/N)^2 + (1.5 * dp/p)^2
# sigma_Y = Y * sqrt( (0.5 * sigma_N / N)^2 + (1.5 * sigma_p / p)^2 )

rel_err_N = 0.5 * sigma_N_fermi / N_corr_fermi
rel_err_p = 1.5 * sigma_pc_fermi / pc_fermi
sigma_Y_fermi = Y_fermi * np.sqrt(rel_err_N**2 + rel_err_p**2)

# --- 5. Линейная аппроксимация для поиска T_max ---
# На графике Ферми бета-спектр - это прямая линия, идущая вниз.
# У Cs-137 основной бета-переход (на 661 кэВ уровень) имеет граничную энергию около 514 кэВ.
# Конверсионный пик (624 кэВ) на графике Ферми будет выглядеть как "горб" на хвосте.
# Нам нужно аппроксимировать ЛИНЕЙНЫЙ участок ДО ПИКА.
# Визуально по данным: от ~200 кэВ до ~450 кэВ.

mask_fit = (T_fermi > 300) & (T_fermi < 500)
T_fit = T_fermi[mask_fit]
Y_fit = Y_fermi[mask_fit]
sigma_Y_fit = sigma_Y_fermi[mask_fit]


def linear_func(x, a, b):
    return a * x + b


popt, pcov = curve_fit(
    linear_func, T_fit, Y_fit, sigma=sigma_Y_fit, absolute_sigma=True
)
a, b = popt
perr = np.sqrt(np.diag(pcov))
print(perr, a, b)
err = np.sqrt((perr[0] / a) ** 2 + (perr[1] / b) ** 2) * (-b / a)
print(err)
# Расчет T_max (точка пересечения с нулем: ax + b = 0 => x = -b/a)
T_max_calc = -b / a

# Погрешность T_max (через производные)
# dT = sqrt( (d(-b/a)/da * da)^2 + (d(-b/a)/db * db)^2 )
# dT = sqrt( (b/a^2 * da)^2 + (-1/a * db)^2 )
sigma_T_max = np.sqrt(((b / a**2) * perr[0]) ** 2 + ((-1 / a) * perr[1]) ** 2)

print(f"--- Результат по графику Ферми ---")
print(f"Граничная энергия (T_max): {T_max_calc:.2f} ± {sigma_T_max:.2f} кэВ")

# Линия для отрисовки
x_line = np.linspace(150, T_max_calc + 50, 100)
y_line = linear_func(x_line, a, b)

# --- 6. Построение графика ---
plt.figure(figsize=(10, 8))

# Точки с крестами погрешностей
plt.errorbar(
    T_fermi,
    Y_fermi,
    xerr=sigma_T_fermi,
    yerr=sigma_Y_fermi,
    fmt="o",
    color="green",
    ecolor="gray",
    elinewidth=1,
    capsize=2,
    markersize=4,
)

# Прямая аппроксимации
plt.plot(x_line, y_line, "r-", linewidth=2, label=f"Линейная аппроксимация")

# Оформление
plt.xlabel(r"$T_e$, кэВ", fontsize=12)
plt.ylabel(r"$\frac{\sqrt{N-N_\phi}}{p^{3/2}} \times 10^6$", fontsize=12)
plt.grid(True, which="major", linestyle="--")
plt.axhline(0, color="black", linewidth=1)

# Линия центра пика
plt.axvline(
    T_max_calc,
    color="red",
    linestyle="--",
    label=f"$T_{{max}} = {T_max_calc:.1f} \pm {sigma_T_max:.1f}$ кэВ",
)

plt.xlim(0, 800)
plt.ylim(bottom=-0.5)

plt.legend()
plt.tight_layout()
plt.show()
