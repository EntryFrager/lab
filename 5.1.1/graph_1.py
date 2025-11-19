import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Данные для всех длин волн
data = {
    5852: {
        "V": np.array(
            [
                7.156,
                5.006,
                2.973,
                2.799,
                2.2,
                1.795,
                1.53,
                1.359,
                1.204,
                1.18,
                1.129,
                1.07,
                1.02,
                0.97,
                0.92,
                0.87,
                0.8,
                0.73,
                0.66,
                0.61,
                0.56,
                0.5,
                0.45,
                0.4,
                0.35,
                0.3,
                0.25,
                0.2,
                0.15,
                0.1,
                0.05,
                0.008,
                -0.1,
                -0.17,
                -0.24,
                -0.34,
                -0.44,
                -0.54,
                -0.66,
                -0.8,
                -1.16,
                -1.505,
                -2.507,
            ]
        ),
        "I": np.array(
            [
                0.582,
                0.556,
                0.51,
                0.505,
                0.476,
                0.445,
                0.415,
                0.387,
                0.356,
                0.353,
                0.342,
                0.327,
                0.314,
                0.303,
                0.288,
                0.273,
                0.253,
                0.233,
                0.213,
                0.199,
                0.184,
                0.168,
                0.156,
                0.143,
                0.132,
                0.119,
                0.108,
                0.097,
                0.086,
                0.076,
                0.066,
                0.058,
                0.052,
                0.041,
                0.032,
                0.023,
                0.018,
                0.015,
                0.012,
                0.012,
                0.011,
                0.01,
                0.008,
            ]
        ),
    },
    6074: {
        "V": np.array(
            [
                7.164,
                5.504,
                4.526,
                3.511,
                2.543,
                1.697,
                1.301,
                1.1,
                1.0,
                0.8,
                0.65,
                0.5,
                0.4,
                0.3,
                0.2,
                0.1,
                0.05,
                0.0,
                -0.1,
                -0.15,
                -0.2,
                -0.25,
                -0.3,
                -0.35,
                -0.4,
                -0.5,
                -0.8,
                -1.2,
                -2.506,
                -3.445,
                -4.25,
                -7.146,
            ]
        ),
        "I": np.array(
            [
                0.596,
                0.578,
                0.564,
                0.542,
                0.514,
                0.462,
                0.42,
                0.379,
                0.354,
                0.291,
                0.241,
                0.192,
                0.162,
                0.132,
                0.106,
                0.082,
                0.071,
                0.061,
                0.054,
                0.045,
                0.038,
                0.033,
                0.028,
                0.024,
                0.023,
                0.019,
                0.017,
                0.016,
                0.012,
                0.012,
                0.011,
                0.011,
            ]
        ),
    },
    6217: {
        "V": np.array(
            [
                7.169,
                6.501,
                5.503,
                4.5,
                3.501,
                2.502,
                1.807,
                1.4,
                1.1,
                0.9,
                0.8,
                0.7,
                0.6,
                0.45,
                0.3,
                0.2,
                0.1,
                0.05,
                0.0,
                -0.1,
                -0.2,
                -0.3,
                -0.45,
                -0.8,
                -1.203,
                -2.017,
                -4.006,
                -7.149,
            ]
        ),
        "I": np.array(
            [
                0.603,
                0.598,
                0.588,
                0.573,
                0.553,
                0.52,
                0.484,
                0.445,
                0.393,
                0.34,
                0.302,
                0.273,
                0.236,
                0.181,
                0.138,
                0.109,
                0.083,
                0.072,
                0.061,
                0.053,
                0.039,
                0.03,
                0.024,
                0.021,
                0.02,
                0.018,
                0.016,
                0.015,
            ]
        ),
    },
    6402: {
        "V": np.array(
            [
                7.169,
                6.506,
                5.002,
                3.525,
                2.503,
                1.808,
                1.402,
                1.1,
                0.8,
                0.6,
                0.4,
                0.2,
                0.1,
                0.0,
                -0.1,
                -0.3,
                -0.5,
                -0.7,
                -1.0,
                -2.0,
            ]
        ),
        "I": np.array(
            [
                0.609,
                0.604,
                0.59,
                0.558,
                0.525,
                0.482,
                0.443,
                0.357,
                0.305,
                0.225,
                0.167,
                0.106,
                0.078,
                0.054,
                0.048,
                0.029,
                0.024,
                0.023,
                0.022,
                0.02,
            ]
        ),
    },
    6717: {
        "V": np.array(
            [
                7.169,
                6.018,
                4.501,
                3.511,
                2.514,
                1.801,
                1.402,
                1.0,
                0.8,
                0.6,
                0.4,
                0.2,
                0.1,
                0.0,
                -0.2,
                -0.3,
                -0.5,
                -0.7,
                -1.0,
                -2.0,
            ]
        ),
        "I": np.array(
            [
                0.614,
                0.607,
                0.586,
                0.562,
                0.509,
                0.453,
                0.402,
                0.312,
                0.262,
                0.201,
                0.139,
                0.086,
                0.062,
                0.051,
                0.047,
                0.043,
                0.041,
                0.04,
                0.04,
                0.04,
            ]
        ),
    },
}


# Линейная функция для аппроксимации
def linear_func(V, k, V0):
    return k * (V - V0)


# Создаем графики для каждой длины волны
results = {}

ranges = {
    5852: (0.02, 0.36),
    6074: (0.0235, 0.35),
    6217: (0.025, 0.35),
    6402: (0.026, 0.35),
    6717: (0.047, 0.35),
}

for wavelength, measurements in data.items():
    V = measurements["V"]
    I = measurements["I"]

    # Вычисляем квадратный корень из тока
    sqrt_I = np.sqrt(I)

    # Выбираем линейный участок (область малых токов)
    I_min, I_max = ranges[wavelength]
    linear_mask = (I > I_min) & (I < I_max)
    V_linear = V[linear_mask]
    sqrt_I_linear = sqrt_I[linear_mask]

    # Аппроксимируем линейный участок
    p0 = [0.1, -1.0]
    popt, pcov = curve_fit(linear_func, V_linear, sqrt_I_linear, p0=p0)
    k, V0 = popt
    sigma_k, sigma_V0 = np.sqrt(np.diag(pcov))

    results[wavelength] = {"k": k, "sigma_k": sigma_k, "V0": V0, "sigma_V0": sigma_V0}

    # Создаем два подграфика для каждой длины волны
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Первый график: I от V
    ax1.plot(V, I, "bo-", markersize=4, linewidth=1, label="I(V)")
    ax1.set_xlabel("Напряжение V, В")
    ax1.set_ylabel("Сила тока I, А")
    ax1.set_title(f"Зависимость I от V для λ = {wavelength} Å")
    ax1.axvline(x=0, color="k", linestyle="-", alpha=0.3)
    ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Второй график: √I от V (только точки для аппроксимации)
    ax2.plot(
        V_linear, sqrt_I_linear, "ro", markersize=4, label="Экспериментальные точки"
    )

    # Аппроксимирующая прямая
    V_fit = np.linspace(min(V_linear) - 0.5, max(V_linear) + 0.5, 100)
    sqrt_I_fit = linear_func(V_fit, k, V0)
    ax2.plot(
        V_fit, sqrt_I_fit, "r-", linewidth=2, label=f"Аппроксимация: √I = k(V - V₀)"
    )

    ax2.axvline(
        x=V0,
        color="g",
        linestyle="--",
        alpha=0.7,
        label=f"V₀ = {V0:.3f} ± {sigma_V0:.3f} В",
    )
    ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax2.axvline(x=0, color="k", linestyle="-", alpha=0.3)

    ax2.set_xlabel("Напряжение V, В")
    ax2.set_ylabel("√I, √A")
    ax2.set_title(f"Зависимость √I от V для λ = {wavelength} Å")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Добавляем текст с коэффициентами на график
    textstr = "\n".join(
        (
            f"$k = {k:.3f} \pm {sigma_k:.3f}$ В$^{{-1}}$",
            f"$V_0 = {V0:.3f} \pm {sigma_V0:.3f}$ В",
        )
    )

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax2.text(
        0.05,
        0.95,
        textstr,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    plt.show()

    print(f"λ = {wavelength} Å:")
    print(f"  Запирающий потенциал V0 = {V0:.3f} ± {sigma_V0:.3f} В")
    print(f"  Коэффициент наклона k = {k:.3f} ± {sigma_k:.3f} В^{-1}")
    print()

# Сводная таблица результатов
print("Сводная таблица результатов:")
print("λ (Å)     V₀ (В)       σ_V₀ (В)     k (В⁻¹)     σ_k (В⁻¹)")
print("-" * 60)
for wavelength in sorted(results.keys()):
    res = results[wavelength]
    print(
        f"{wavelength:4}    {res['V0']:8.3f}    {res['sigma_V0']:8.3f}    {res['k']:8.3f}    {res['sigma_k']:8.3f}"
    )
