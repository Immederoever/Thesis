import matplotlib.pyplot as plt
import numpy as np

categories = ['HC top-1', 'LC top-1', 'HC top-5', 'LC top-5', 'HC top-10', 'LC top-10']

child_scores = [18.9, 0.0, 28.9, 2.2, 35.6, 4.4]
adult_scores = [34.4, 3.3, 48.9, 8.9, 54.4, 14.4]
gpt_scores   = [51.1, 7.8, 76.7, 13.3, 78.9, 21.1]
xl_scores    = [67.8, 10.0, 90.0, 24.4, 91.1, 37.8]

x = np.arange(len(categories))
width = 0.20

fig, ax = plt.subplots(figsize=(15, 6))

rects1 = ax.bar(x - 1.5*width, child_scores, width, label='Child model', color='#add8e6', edgecolor='black')
rects2 = ax.bar(x - 0.5*width, adult_scores, width, label='Adult model', color='#4682b4', edgecolor='black')
rects3 = ax.bar(x + 0.5*width, gpt_scores, width, label='GPT-2', color='#c44e52', edgecolor='black')
rects4 = ax.bar(x + 1.5*width, xl_scores, width, label='GPT-2 XL', color='#dd8452', edgecolor='black')

ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_title("Accuracies of this project's models next to the fully trained GPT-2 and GPT-2 XL models", fontsize=17, pad=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

ax.axvline(x=1.5, color='gray', linestyle=':', alpha=0.6)
ax.axvline(x=3.5, color='gray', linestyle=':', alpha=0.6)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}', 
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", 
                    ha='center', fontsize=13, weight='bold', rotation=0)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

plt.tight_layout()
plt.savefig("plot_together.png", dpi=300)
print("Saved plot_together.png")
