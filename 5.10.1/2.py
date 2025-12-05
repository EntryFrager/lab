import matplotlib.pyplot as plt
import numpy as np

# Ваши данные
nu = np.array([110, 120, 140, 150, 160])  # Частота, МГц
A = np.array([26, 30, 36, 38, 41])  # Расстояние между пиками, дел.

# Линейная аппроксимация с расчетом ковариационной матрицы (cov=True)
# p - коэффициенты (k, b), V - ковариационная матрица
p, V = np.polyfit(A, nu, 1, cov=True)

k = p[0]  # Наклон
b = p[1]  # Свободный член

# Погрешности - это квадратные корни из диагональных элементов матрицы ковариации
sigma_k = np.sqrt(V[0, 0])
sigma_b = np.sqrt(V[1, 1])

# Расчет R^2
correlation_matrix = np.corrcoef(A, nu)
correlation_xy = correlation_matrix[0, 1]
r_squared = correlation_xy**2

print(f"--- РЕЗУЛЬТАТЫ РАСЧЕТА ---")
print(f"Уравнение: f = ({k:.2f} ± {sigma_k:.2f}) * A + ({b:.2f} ± {sigma_b:.2f})")
print(f"Коэффициент наклона k = {k:.2f} ± {sigma_k:.2f} МГц/дел")
print(f"R^2 = {r_squared:.4f}")

# Создание линии тренда
x_fit = np.linspace(min(A) - 0.5, max(A) + 0.5, 100)
y_fit = k * x_fit + b

# Построение графика
plt.figure(figsize=(10, 7), dpi=200)

plt.scatter(A, nu, color="darkblue", s=80, label="Экспериментальные точки", zorder=5)
plt.plot(
    x_fit,
    y_fit,
    "r--",
)

plt.xlabel("$A$, дел.", fontsize=12)
plt.ylabel("$f$, МГц", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=12)

plt.show()
