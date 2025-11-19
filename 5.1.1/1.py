import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Пример данных
num_votes = [1, 5, 10, 15, 20]
w_f1 = [0.65, 0.78, 0.82, 0.84, 0.85]
ua = [0.62, 0.75, 0.79, 0.81, 0.82]
wa = [0.67, 0.80, 0.83, 0.85, 0.86]

# Создание комплексного отчета
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Основной линейный график
ax1.plot(num_votes, w_f1, "o-", label="Weighted F1", linewidth=2)
ax1.plot(num_votes, ua, "s-", label="Unweighted Accuracy", linewidth=2)
ax1.plot(num_votes, wa, "^-", label="Weighted Accuracy", linewidth=2)
ax1.set_xlabel("Number of Votes")
ax1.set_ylabel("Score")
ax1.set_title("Основные метрики")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(num_votes)

# 2. График улучшений (процентное изменение от базового)
improvement_f1 = [(x / w_f1[0] - 1) * 100 for x in w_f1]
improvement_ua = [(x / ua[0] - 1) * 100 for x in ua]
improvement_wa = [(x / wa[0] - 1) * 100 for x in wa]

ax2.bar(
    [x - 1 for x in num_votes],
    improvement_f1,
    width=0.8,
    label="Weighted F1",
    alpha=0.7,
)
ax2.bar(num_votes, improvement_ua, width=0.8, label="Unweighted Accuracy", alpha=0.7)
ax2.bar(
    [x + 1 for x in num_votes],
    improvement_wa,
    width=0.8,
    label="Weighted Accuracy",
    alpha=0.7,
)
ax2.set_xlabel("Number of Votes")
ax2.set_ylabel("Improvement (%)")
ax2.set_title("Процентное улучшение от baseline")
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Heatmap корреляций
corr_data = np.array([w_f1, ua, wa])
corr_matrix = np.corrcoef(corr_data)

im = ax3.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
ax3.set_xticks([0, 1, 2])
ax3.set_yticks([0, 1, 2])
ax3.set_xticklabels(["W-F1", "U-Acc", "W-Acc"])
ax3.set_yticklabels(["W-F1", "U-Acc", "W-Acc"])
ax3.set_title("Матрица корреляций")

# Добавление значений в heatmap
for i in range(3):
    for j in range(3):
        ax3.text(
            j,
            i,
            f"{corr_matrix[i, j]:.3f}",
            ha="center",
            va="center",
            fontweight="bold",
        )

# 4. Статистика сходимости
convergence_f1 = [abs(w_f1[i] - w_f1[i - 1]) if i > 0 else 0 for i in range(len(w_f1))]
convergence_ua = [abs(ua[i] - ua[i - 1]) if i > 0 else 0 for i in range(len(ua))]
convergence_wa = [abs(wa[i] - wa[i - 1]) if i > 0 else 0 for i in range(len(wa))]

ax4.plot(num_votes[1:], convergence_f1[1:], "o-", label="Weighted F1")
ax4.plot(num_votes[1:], convergence_ua[1:], "s-", label="Unweighted Accuracy")
ax4.plot(num_votes[1:], convergence_wa[1:], "^-", label="Weighted Accuracy")
ax4.set_xlabel("Number of Votes")
ax4.set_ylabel("Absolute Change")
ax4.set_title("Скорость сходимости (изменение между шагами)")
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xticks(num_votes[1:])

plt.tight_layout()
plt.show()

# Вывод статистического отчета
print("=" * 60)
print("СТАТИСТИЧЕСКИЙ ОТЧЕТ")
print("=" * 60)
print(f"Максимальные значения:")
print(f"  Weighted F1: {max(w_f1):.3f} при {num_votes[w_f1.index(max(w_f1))]} голосах")
print(
    f"  Unweighted Accuracy: {max(ua):.3f} при {num_votes[ua.index(max(ua))]} голосах"
)
print(f"  Weighted Accuracy: {max(wa):.3f} при {num_votes[wa.index(max(wa))]} голосах")
print(f"\nОбщее улучшение:")
print(f"  Weighted F1: +{max(w_f1)-w_f1[0]:.3f} ({((max(w_f1)/w_f1[0])-1)*100:.1f}%)")
print(f"  Unweighted Accuracy: +{max(ua)-ua[0]:.3f} ({((max(ua)/ua[0])-1)*100:.1f}%)")
print(f"  Weighted Accuracy: +{max(wa)-wa[0]:.3f} ({((max(wa)/wa[0])-1)*100:.1f}%)")
