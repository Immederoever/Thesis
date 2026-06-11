import matplotlib.pyplot as plt
import numpy as np

# Data: [HC top1, LC top1, HC top-5, LC top-5, HC top-10, LC top-10]
child_scores = [17.8, 1.1, 32.2, 4.4, 35.6, 7.8]
adult_scores = [36.7, 3.3, 50.0, 12.2, 58.9, 12.2]

categories = ['HC top-1', 'LC top-1', 'HC top-5', 'LC top-5', 'HC top-10', 'LC top-10']
x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 5))

rects1 = ax.bar(x - width/2, child_scores, width, label='Child model', color='#add8e6', edgecolor='black')
rects2 = ax.bar(x + width/2, adult_scores, width, label='Adult model', color='#4682b4', edgecolor='black')

ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_title('Comparison of accuracies between the child and adult model', fontsize=17, pad=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=14)
ax.set_ylim(0, 65) # Range of vertical axis
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(loc='upper left', fontsize=13)

# Vertical lines seperating categories
ax.axvline(x=1.5, color='gray', linestyle=':', alpha=0.6)
ax.axvline(x=3.5, color='gray', linestyle=':', alpha=0.6)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=12, weight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig("plot_t1_t5_t10_OPT.png", dpi=300)
print("Saved plot_t1_t5_t10_OPT.png")
