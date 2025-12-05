import numpy as np
import matplotlib.pyplot as plt

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

# Время измерения
tau = 100.0

# --- 2. Расчет погрешностей ---
# Погрешность скорости счета (статистическая): sigma_N = sqrt(N / tau)
# Важно: берем "сырое" N, так как статистика работает для исходных отсчетов
sigma_N = np.sqrt(N / tau)

# Погрешность тока (sigma_I).
# Если прибор цифровой, обычно это 1 единица младшего разряда или указано в паспорте.
# Для визуализации "креста" возьмем, например, 0.02 А (или поставь 0, если не нужно по X)
sigma_I = 0.02 * np.ones_like(I)

# --- 3. Учет фона ---
# N_bg = slope * I + intercept
slope = (N[-1] - N[0]) / (I[-1] - I[0])
intercept = N[0]
background = slope * I + intercept
N_corrected = N - background

# --- 4. Выбор диапазона и ВЗВЕШЕННАЯ аппроксимация ---
# Диапазон пика конверсии
idx_start = np.where(I == 4.05)[0][0]
idx_end = np.where(I == 4.45)[0][0]

I_fit = I[idx_start : idx_end + 1]
N_fit = N_corrected[idx_start : idx_end + 1]
sigma_N_fit = sigma_N[idx_start : idx_end + 1]
# --- 4. Выбор диапазона и ВЗВЕШЕННАЯ аппроксимация ---
# ... (код выбора диапазонов остается тем же) ...

weights = 1 / (sigma_N_fit**2)

# Добавляем cov=True, чтобы получить ковариационную матрицу
coeffs, cov_matrix = np.polyfit(I_fit, N_fit, 2, w=weights, cov=True)
a, b, c = coeffs

# --- 5. Определение центра пика и ЕГО ПОГРЕШНОСТИ ---
I_peak = -b / (2 * a)

# Извлекаем дисперсии и ковариацию из матрицы
# cov_matrix - это матрица 3x3 (для a, b, c)
var_a = cov_matrix[0, 0]  # дисперсия a (sigma_a^2)
var_b = cov_matrix[1, 1]  # дисперсия b (sigma_b^2)
cov_ab = cov_matrix[0, 1]  # ковариация между a и b

# Частные производные
dI_da = b / (2 * a**2)
dI_db = -1 / (2 * a)

# Итоговая формула переноса ошибок
sigma_I_peak = np.sqrt(
    (dI_da**2 * var_a) + (dI_db**2 * var_b) + (2 * dI_da * dI_db * cov_ab)
)

print("-" * 30)
print(f"Уравнение параболы: N(I) = {a:.3f}*I^2 + {b:.3f}*I + {c:.3f}")
print(f"Ток пика конверсии: I_peak = {I_peak:.4f} +/- {sigma_I_peak:.4f} А")
print("-" * 30)

# Для графика ничего менять не нужно, только добавь вывод в легенду или заголовок:
plt.axvline(
    I_peak,
    color="red",
    linestyle="--",
    label=f"$I_{{peak}} = {I_peak:.3f} \pm {sigma_I_peak:.3f}$ A",
)

# --- 6. Построение графика ---
I_range = np.linspace(I_fit.min(), I_fit.max(), 100)
N_fit_curve = a * I_range**2 + b * I_range + c

plt.figure(figsize=(10, 7))

# Основные точки с "крестами" погрешностей
# fmt='o' — точки, ecolor — цвет планок, capsize — "шляпки" у планок
plt.errorbar(
    I,
    N_corrected,
    yerr=sigma_N,
    xerr=sigma_I,
    fmt="o",
    color="green",
    ecolor="gray",
    capsize=3,
)

# Аппроксимация
plt.plot(I_range, N_fit_curve, "r-", linewidth=2, label="Параболическая аппроксимация")

# Линия центра пика
plt.axvline(I_peak, color="red", linestyle="--", label=f"$I_{{peak}} = {I_peak:.3f}$ А")

# Подписи и сетка
plt.xlabel("$I$, А")
plt.ylabel("$N$, 1/с")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()
