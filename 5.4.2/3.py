import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- 1. Входные данные ---
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

# --- 2. Константы и Погрешности параметров ---
tau = 100.0  # Время измерения, с
k = 238.0  # Калибровка, кэВ/(с*А)
sigma_k = 1.0  # Погрешность калибровки
sigma_I_const = 0.02  # Погрешность считывания тока
m0c2 = 511.0  # Энергия покоя, кэВ

# --- 3. Расчет Энергии и Импульса ---
pc = I * k
E_total = np.sqrt(pc**2 + m0c2**2)
T = E_total - m0c2  # Кинетическая энергия

# --- 4. Расчет Погрешностей (Error bars) ---
# Погрешность N (статистическая)
sigma_N = np.sqrt(N_measured / tau)
sigma_N[sigma_N == 0] = 1.0 / tau  # Защита от деления на ноль, если вдруг 0 отсчетов

# Погрешность T (косвенная)
sigma_pc = np.sqrt((k * sigma_I_const) ** 2 + (I * sigma_k) ** 2)
dT_dpc = pc / E_total
sigma_T = np.abs(dT_dpc) * sigma_pc

# --- 5. Вычитание фона (Метод прямой линии через края) ---
slope_bg = (N_measured[-1] - N_measured[0]) / (I[-1] - I[0])
intercept_bg = N_measured[0] - slope_bg * I[0]

N_background = slope_bg * I + intercept_bg
N_corrected = N_measured - N_background

# --- 6. Аппроксимация пика (В указанном диапазоне) ---


# Модель: Парабола в вершинной форме: y = a*(x - x0)^2 + h
# x0 - это сразу положение пика (энергия), h - высота
def parabola_vertex(x, a, x0, h):
    return a * (x - x0) ** 2 + h


# !! ВАЖНО: Выбор диапазона по току от 4.05 до 4.45 А !!
mask_peak = (I >= 4.05) & (I <= 4.45)

T_peak_data = T[mask_peak]
N_peak_data = N_corrected[mask_peak]
sigma_N_peak = sigma_N[mask_peak]

# Начальное приближение [кривизна, примерный центр, высота]
# Центр берем примерно 624 кэВ (теоретический), высота ~8
p0 = [-0.1, 624.0, 8.0]

# Фиттинг с учетом весов (погрешностей по Y)
popt, pcov = curve_fit(
    parabola_vertex,
    T_peak_data,
    N_peak_data,
    sigma=sigma_N_peak,
    absolute_sigma=True,
    p0=p0,
)

a_fit, T_peak_fit, h_fit = popt

# Извлекаем погрешность определения вершины пика
# pcov - ковариационная матрица, диагональные элементы - дисперсии параметров
perr = np.sqrt(np.diag(pcov))
sigma_T_peak_fit = perr[1]  # Ошибка параметра x0 (энергии пика)

print(f"=" * 40)
print(f"РЕЗУЛЬТАТЫ ОБРАБОТКИ КОНВЕРСИОННОГО ПИКА")
print(f"Диапазон тока для фита: 4.05 - 4.45 А")
print(f"Максимум энергии (из фита): {T_peak_fit:.2f} кэВ")
print(f"Погрешность определения максимума: ± {sigma_T_peak_fit:.2f} кэВ")
print(f"=" * 40)

# Данные для отрисовки плавной красной линии аппроксимации
T_smooth = np.linspace(min(T_peak_data), max(T_peak_data), 100)
N_smooth = parabola_vertex(T_smooth, *popt)

# --- 7. Построение графика ---
plt.figure(figsize=(10, 7))

# Экспериментальные точки
plt.errorbar(
    T,
    N_corrected,
    xerr=sigma_T,
    yerr=sigma_N,
    fmt="o",
    color="green",
    ecolor="gray",
    elinewidth=1,
    capsize=3,
)

# Линия аппроксимации
plt.plot(T_smooth, N_smooth, "r-", linewidth=2.5, label="Аппроксимация (парабола)")

# Линия центра пика
plt.axvline(
    T_peak_fit,
    color="red",
    linestyle="--",
    label=f"$T_{{peak}} = {T_peak_fit:.1f} \pm {sigma_T_peak_fit:.1f}$ кэВ",
)

# Оформление осей и сетки
plt.xlabel(r"$T_e$, кэВ", fontsize=12)
plt.ylabel(r"$N$, $c^{-1}$", fontsize=12)
plt.grid(True, which="major", linestyle="--", alpha=0.7)
plt.axhline(0, color="black", linewidth=0.8)  # Ось нуля

plt.xlim(0, 1300)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
