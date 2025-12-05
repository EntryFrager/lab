import numpy as np
import matplotlib.pyplot as plt

# 1. Данные из таблицы
U_R = np.array([5.2, 15.1, 25.1, 35.1, 45.1, 55.4, 65.5, 76.4, 85.4, 95.1])
E_front = np.array([0.87, 2.61, 4.34, 6.12, 7.87, 9.63, 11.43, 13.40, 14.99, 16.81])
E_back = np.array([0.96, 2.89, 4.70, 6.47, 8.32, 10.30, 12.04, 13.91, 15.71, 17.37])

# 2. Расчет средних значений и погрешностей
E_avg = (E_front + E_back) / 2

# Инструментальная погрешность
sigma_instr = 0.04

# Погрешность усреднения (половина разности между передним и задним замером)
sigma_spread = np.abs(E_back - E_front) / 2

# Полная погрешность E (корень из суммы квадратов)
sigma_E = np.sqrt(sigma_instr**2 + sigma_spread**2)

# Погрешность U (только инструментальная)
sigma_U = np.full_like(U_R, sigma_instr)

# 3. Линейная аппроксимация (y = kx + b)
# Учитываем веса точек как 1/sigma^2 для более точного фита, или просто МНК
# Здесь используем обычный np.polyfit для простоты, так как зависимость очень линейная
k, b = np.polyfit(U_R, E_avg, 1)

print(f"Угловой коэффициент k = {k:.4f}")

# 4. Построение графика
plt.figure(figsize=(12, 10), dpi=200)

# Отрисовка точек с крестами погрешностей
# xerr - погрешность по X, yerr - погрешность по Y
# capsize - размер "шляпок" на усах погрешности
plt.errorbar(
    U_R,
    E_avg,
    xerr=sigma_U,
    yerr=sigma_E,
    fmt="ro",
    ecolor="black",
    capsize=3,
    zorder=5,
)

# Отрисовка линии аппроксимации
x_fit = np.linspace(0, 100, 100)
y_fit = k * x_fit + b
plt.plot(x_fit, y_fit, "b--")

plt.xlabel("$U_R$, мВ", fontsize=12)
plt.ylabel("$\mathcal{E}_{сред}$, мВ", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=12)

# Сохранение
plt.show()
