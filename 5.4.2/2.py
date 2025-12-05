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

# --- 2. Константы и Параметры ---
tau = 100.0  # Время измерения, с
k = 238.0  # Коэффициент калибровки, кэВ/(с*А)
sigma_k = 1.0  # Погрешность коэффициента, кэВ/(с*А)
sigma_I_const = 0.02  # Абсолютная погрешность тока, А

# --- 3. Расчет импульсов и погрешностей ---

# 3.1. Импульс
pc = I * k

# 3.2. Погрешность по Y (N): Статистическая
# sigma_N = sqrt(N_raw) / tau = sqrt(N_measured * tau) / tau = sqrt(N_measured / tau)
sigma_N = np.sqrt(N_measured / tau)

# 3.3. Погрешность по X (pc): Косвенная погрешность произведения
# sigma_p = sqrt( (k * sigma_I)^2 + (I * sigma_k)^2 )
sigma_pc = np.sqrt((k * sigma_I_const) ** 2 + (I * sigma_k) ** 2)

# --- 4. Вычитание фона ---
# Прямая через первую (0 А) и последнюю (5 А) точки
slope_bg = (N_measured[-1] - N_measured[0]) / (I[-1] - I[0])
intercept_bg = N_measured[0]
N_background = slope_bg * I + intercept_bg
N_corrected = N_measured - N_background

# --- 5. Взвешенная аппроксимация пика (в координатах Тока I) ---
# Мы ищем параметры параболы для тока, так как так точнее,
# а потом переведем результат в импульс.

# Диапазон аппроксимации (возьмем точки вокруг пика 4.05 - 4.45 А)
idx_start = np.where(I == 4.05)[0][0]
idx_end = np.where(I == 4.45)[0][0]

I_fit = I[idx_start : idx_end + 1]
N_fit = N_corrected[idx_start : idx_end + 1]
sigma_N_fit = sigma_N[idx_start : idx_end + 1]

# Веса для МНК = 1 / sigma^2
weights = 1 / (sigma_N_fit**2)

# Аппроксимация полиномом 2-й степени: N(I) = a*I^2 + b*I + c
coeffs, cov_matrix = np.polyfit(I_fit, N_fit, 2, w=weights, cov=True)
a, b, c = coeffs

# --- 6. Расчет положения пика и его погрешности ---

# Положение пика по току: I = -b / 2a
I_peak = -b / (2 * a)

# Погрешность положения пика по току (sigma_I_peak) через ковариационную матрицу
var_a = cov_matrix[0, 0]
var_b = cov_matrix[1, 1]
cov_ab = cov_matrix[0, 1]

dI_da = b / (2 * a**2)
dI_db = -1 / (2 * a)

sigma_I_peak_stat = np.sqrt(
    (dI_da**2 * var_a) + (dI_db**2 * var_b) + (2 * dI_da * dI_db * cov_ab)
)

# Перевод пика в импульс (pc)
pc_peak = I_peak * k

# Полная погрешность импульса в пике.
# Складываем погрешность определения вершины (статистика фиттинга + приборная тока)
# и погрешность коэффициента k.
# sigma_I_total = sqrt(sigma_I_peak_stat^2 + sigma_I_const^2) - полная ошибка тока в пике
sigma_I_total = np.sqrt(sigma_I_peak_stat**2 + sigma_I_const**2)

# sigma_pc_peak = sqrt( (k * sigma_I_total)^2 + (I_peak * sigma_k)^2 )
sigma_pc_peak = np.sqrt((k * sigma_I_total) ** 2 + (I_peak * sigma_k) ** 2)

print("-" * 40)
print(f"Координата пика (Ток): {I_peak:.4f} А")
print(f"Координата пика (Импульс): {pc_peak:.2f} кэВ/c")
print(f"Погрешность импульса в пике: +/- {sigma_pc_peak:.2f} кэВ/c")
print("-" * 40)

# --- 7. Подготовка кривой аппроксимации для графика ---
# Генерируем точки тока для гладкой кривой
I_smooth = np.linspace(I_fit.min(), I_fit.max(), 100)
# Считаем значения параболы
N_smooth = a * I_smooth**2 + b * I_smooth + c
# Переводим ток в импульс для оси X
pc_smooth = I_smooth * k

# --- 8. Построение графика ---
plt.figure(figsize=(10, 7))

# Экспериментальные данные с крестами ошибок
plt.errorbar(
    pc,
    N_corrected,
    yerr=sigma_N,
    xerr=sigma_pc,
    fmt="o",
    color="black",
    ecolor="gray",
    elinewidth=1,
    capsize=2,
    markersize=4,
)

# Кривая аппроксимации
plt.plot(pc_smooth, N_smooth, "r-", linewidth=2, label="Параболическая аппроксимация")

# Вертикальная линия пика
plt.axvline(
    pc_peak,
    color="red",
    linestyle="--",
    alpha=0.7,
    label=f"$p_{{peak}} = {pc_peak:.1f} \pm {sigma_pc_peak:.1f}$ кэВ/c",
)


plt.xlabel(r"$p_e$, кэВ/с", fontsize=12)
plt.ylabel(r"$N$, $c^{-1}$", fontsize=12)
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.axhline(0, color="black", linewidth=0.8)
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()
