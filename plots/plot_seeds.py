import matplotlib.pyplot as plt
import numpy as np

categories = ['Child model HC', 'Adult model HC', 'Child model LC', 'Adult model LC']

data_seed_0   = [17.8, 36.7, 1.1, 3.3]
data_seed_1   = [17.8, 33.3, 0.0, 3.3]
data_seed_42  = [18.9, 34.4, 0.0, 3.3]
data_seed_123 = [17.8, 32.2, 0.0, 1.1]
data_seed_444 = [16.7, 30.0, 1.1, 3.3]

all_data = [data_seed_0, data_seed_1, data_seed_42, data_seed_123, data_seed_444]
seed_labels = ['Seed 0', 'Seed 1', 'Seed 42', 'Seed 123', 'Seed 444']

x = np.arange(len(categories))
width = 0.15
offsets = [-2*width, -width, 0, width, 2*width]

fig, ax = plt.subplots(figsize=(14, 5))

colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3']

for i in range(5):
    bars = ax.bar(x + offsets[i], all_data[i], width, label=seed_labels[i], color=colors[i], edgecolor='white', linewidth=0.5)

ax.set_ylabel('Top-1 accuracy (%)', fontsize=14)
ax.set_title('Top-1 accuracy from models using different random seeds', fontsize=17, pad=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=14)
ax.set_ylim(0, 42)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(title='Model configurations', fontsize=14, title_fontsize=15, loc='upper right')
fig.tight_layout()
plt.savefig("seeds.png", dpi=300)
print("Saved seeds.png")
