import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# данные
lambda_ne = np.array(
    [
        7032,
        6929,
        6717,
        6678,
        6599,
        6533,
        6507,
        6402,
        6383,
        6334,
        6305,
        6267,
        6217,
        6164,
        6143,
        6096,
        6074,
        6030,
        5976,
        5945,
        5882,
        5852,
        5401,
        5341,
        5331,
    ]
)
phi_ne = np.array(
    [
        2575,
        2546,
        2480,
        2468,
        2442,
        2420,
        2411,
        2374,
        2364,
        2348,
        2335,
        2320,
        2301,
        2278,
        2270,
        2250,
        2240,
        2220,
        2195,
        2182,
        2151,
        2136,
        1878,
        1837,
        1828,
    ]
)

lambda_hg = np.array([6907, 6234, 5791, 5770, 5461, 4916, 4358, 4047])
phi_hg = np.array([2539, 2307, 2105, 2094, 1916, 1497, 835, 280])

# погрешность угла
sigma_phi = 1.4  # градусы


# модель
def model(lmbd, phi0, c, lmbd0):
    return phi0 + c / (lmbd - lmbd0)


# объединённые данные для фитинга
lambda_all = np.concatenate([lambda_ne, lambda_hg])
phi_all = np.concatenate([phi_ne, phi_hg])

# фит
params, cov = curve_fit(model, lambda_all, phi_all, p0=[2000, 1e6, 3000])
phi0_fit, c_fit, lmbd0_fit = params

print("phi0 =", phi0_fit)
print("c =", c_fit)
print("lambda0 =", lmbd0_fit)

# погрешности параметров
sigma_phi0 = np.sqrt(cov[0, 0])
sigma_c = np.sqrt(cov[1, 1])
sigma_lambda0 = np.sqrt(cov[2, 2])

print(f"phi0 = {phi0_fit:.8f} ± {sigma_phi0:.8f} °")
print(f"c = {c_fit:.8f} ± {sigma_c:.8f}")
print(f"lambda0 = {lmbd0_fit:.8f} ± {sigma_lambda0:.8f} Å")

# построение аппроксимации
lmbd_fit = np.linspace(4000, 7500, 1000)
phi_fit = model(lmbd_fit, *params)

# график
plt.figure(figsize=(10, 6))

plt.errorbar(
    lambda_ne, phi_ne, yerr=sigma_phi, fmt="o", markersize=6, capsize=3, label="Неон"
)
plt.errorbar(
    lambda_hg, phi_hg, yerr=sigma_phi, fmt="s", markersize=6, capsize=3, label="Ртуть"
)

plt.plot(lmbd_fit, phi_fit, "k--", label="Аппроксимация")

plt.xlabel("Длина волны, Å")
plt.ylabel("Угол поворота, °")
plt.title("Калибровочная зависимость φ(λ)")
plt.grid(True)
plt.legend()
# plt.gca().invert_yaxis()  # шкала барабана убывает с ростом λ
plt.show()
