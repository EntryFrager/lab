import matplotlib.pyplot as plt
import numpy as np

# 1. Ввод данных
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

N = np.array(
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

# 2. Расчет погрешностей
tau = 100  # время измерения
sigma_N = np.sqrt(N / tau)  # погрешность скорости счета
sigma_I = 0.02  # постоянная погрешность тока

# 3. Расчет фона (линейная интерполяция между первой и последней точкой)
# Уравнение прямой: N = k*I + b
# Первая точка: (I[0], N[0]) -> (0.00, 0.480)
# Последняя точка: (I[-1], N[-1]) -> (5.00, 0.340)

delta_N = N[-1] - N[0]
delta_I = I[-1] - I[0]
slope = delta_N / delta_I  # наклон (k)
intercept = N[0]  # смещение (b)

# Массив значений фона для каждого тока
background = slope * I + intercept

# 4. Вычитание фона
N_corrected = N - background

# --- ПОСТРОЕНИЕ ГРАФИКОВ ---

# Настройка общего стиля
plt.rcParams.update({"font.size": 12})

# График 1: Исходный спектр с линией фона
plt.figure(figsize=(10, 6))
plt.errorbar(
    I,
    N,
    xerr=sigma_I,
    yerr=sigma_N,
    fmt="o",
    markersize=4,
    capsize=3,
    label="Экспериментальные точки",
    color="blue",
    ecolor="black",
)
plt.plot(I, background, "r--", linewidth=2, label="Аппроксимация фона")
plt.xlabel("I, А")
plt.ylabel("N, 1/с")
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# График 2: Спектр с вычтенным фоном
plt.figure(figsize=(10, 6))
# Добавляем горизонтальную линию на нуле
plt.axhline(0, color="gray", linestyle="-", linewidth=1)
plt.errorbar(
    I,
    N_corrected,
    xerr=sigma_I,
    yerr=sigma_N,
    fmt="o",
    markersize=4,
    capsize=3,
    label="N без фона",
    color="green",
    ecolor="black",
)
plt.xlabel("I, А")
plt.ylabel("$N_{corr}$, 1/с")
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
