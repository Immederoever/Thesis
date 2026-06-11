# plot_perplexity.py
import os
import json
import math
import matplotlib.pyplot as plt

OUTPUT_DIR = "FINAL_OPTIMIZED_checkpoints"
CHILD_JSON = os.path.join(OUTPUT_DIR, "stage_child_metrics.json")
ADULT_JSON = os.path.join(OUTPUT_DIR, "stage_adult_metrics.json")
SAVE_PATH = "perplexity.png"

def extract_val_loss(json_path, start=0):
    with open(json_path, 'r') as file:
        logs = json.load(file)
        
    epochs, perplexities = [], []
    
    for log in logs:
        if 'eval_loss' in log and 'epoch' in log:
            epochs.append(log['epoch'] + start)
            perplexities.append(math.exp(log['eval_loss'])) # Converting cross-entropy loss into perplexity
            
    return epochs, perplexities

def main():
    child_epochs, child_perplexities = extract_val_loss(CHILD_JSON, start=0)
    max_child_epoch = max(child_epochs)
    adult_epochs, adult_perplexities = extract_val_loss(ADULT_JSON, start=max_child_epoch)

    plt.figure(figsize=(12, 6), dpi=300)
    
    plt.plot(child_epochs, child_perplexities, label="Child model", color="tab:blue", marker="o", linewidth=2)
    plt.plot(adult_epochs, adult_perplexities, label="Adult model", color="tab:green", marker="s", linewidth=2)

    plt.axvline(x=max_child_epoch, color='gray', linestyle='--', alpha=0.7, label="Stage transition")

    plt.title("Validation perplexity over epochs", fontsize=17, fontweight="bold")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Perplexity", fontsize=14)
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=13)
    
    total_epochs = child_epochs + adult_epochs
    plt.xticks(range(1, int(max(total_epochs)) + 1))
    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"Saved {SAVE_PATH}")

if __name__ == "__main__":
    main()
