import numpy as np
import matplotlib.pyplot as plt

# --- Данные ---
U = np.array(
    [
        8.367,
        7.632,
        7.220,
        7.022,
        6.866,
        6.629,
        8.584,
        8.721,
        9.005,
        9.271,
        9.403,
        9.652,
        9.870,
        10.028,
        10.283,
        10.808,
        8.486,
        6.467,
        6.268,
        6.003,
        5.780,
        5.611,
        5.398,
        5.180,
        5.016,
        4.832,
        4.624,
        4.422,
        4.222,
        4.075,
        3.753,
        3.498,
        3.248,
        3.013,
        2.812,
        2.604,
        2.447,
        2.270,
        2.072,
        1.842,
        1.596,
        1.366,
        0.902,
        0.517,
        0.783,
        0.921,
        1.062,
        1.109,
    ]
)

I = np.array(
    [
        107.82,
        92.20,
        87.19,
        84.99,
        83.29,
        81.45,
        111.09,
        114.45,
        122.49,
        128.61,
        131.68,
        142.01,
        155.01,
        165.60,
        178.03,
        196.34,
        111.74,
        83.07,
        81.98,
        80.98,
        80.53,
        80.44,
        80.69,
        81.25,
        82.00,
        83.09,
        84.78,
        86.84,
        89.37,
        91.56,
        97.47,
        103.47,
        110.67,
        119.02,
        127.70,
        138.95,
        148.98,
        161.94,
        178.09,
        194.76,
        198.01,
        171.23,
        44.07,
        1.59,
        20.21,
        50.30,
        91.30,
        105.81,
    ]
)

# --- Погрешности ---
sigma_U = 0.01
sigma_I = 1.0

# --- Поиск экстремумов ---
extrema = []
for i in range(1, len(I) - 1):
    if I[i] < I[i - 1] and I[i] < I[i + 1]:
        extrema.append(("min", U[i], I[i], i))
    if I[i] > I[i - 1] and I[i] > I[i + 1]:
        extrema.append(("max", U[i], I[i], i))

print("Найденные экстремумы:")
for t, u, i_val, idx in extrema:
    print(f"{t:4s}  U = {u:.3f} В,  I = {i_val:.2f} мкА,  index = {idx}")

# --- Построение графика ---
plt.figure(figsize=(10, 6))
plt.errorbar(
    U, I, xerr=sigma_U, yerr=sigma_I, fmt="o", markersize=4, capsize=3, linestyle="none"
)

# # Отметить экстремумы
# for t, u, i_val, idx in extrema:
#     color = "red" if t == "min" else "green"
#     plt.scatter([u], [i_val], color=color, s=80, label=f"{t} at {u:.2f} V")

plt.xlabel("U, В")
plt.ylabel("I, мкА")
plt.title("ВАХ тиратрона (статический режим)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
