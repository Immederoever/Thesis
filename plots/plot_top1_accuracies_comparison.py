import matplotlib.pyplot as plt
import numpy as np

categories = ['High cloze category', 'Low cloze category']

child_model_scores = [17.8, 1.1]
adult_model_scores = [36.7, 3.3]
child_human_scores = [56.0, 12.0]
adult_human_scores = [85.0, 22.0]

x = np.arange(len(categories))
width = 0.2
gap = 0.05

fig, ax = plt.subplots(figsize=(10, 5))

# Plot humans
rects1 = ax.bar(x - 1.5*width - gap, child_human_scores, width,
                label='Human children', color='#98fb98', edgecolor='black', alpha=0.9)
rects2 = ax.bar(x - 0.5*width - gap, adult_human_scores, width,
                label='Human adults', color='#228b22', edgecolor='black', alpha=0.9)

# Plot models
rects3 = ax.bar(x + 0.5*width + gap, child_model_scores, width,
                label='Child model', color='#add8e6', edgecolor='black', alpha=0.9)
rects4 = ax.bar(x + 1.5*width + gap, adult_model_scores, width,
                label='Adult model', color='#4682b4', edgecolor='black', alpha=0.9)

ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_title('Accuracy of the human participants compared to the language models', 
             fontsize=17, pad=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=14)
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.6)

legend = ax.legend(frameon=True, facecolor='white', edgecolor='gray', loc='upper right', fontsize=14)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, weight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

# Vertical line dividing cloze categories
ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig("top1_accuracies_comparison.png", dpi=300)
print("Saved top1_accuracies_comparison.png")
