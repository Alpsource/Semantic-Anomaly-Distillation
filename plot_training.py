import matplotlib.pyplot as plt
import numpy as np

# Data extracted from training logs!!!
epochs = range(1, 16)

# Accuracy Data
base_acc = [71.01, 83.72, 88.83, 91.42, 93.04, 94.10, 94.91, 95.51, 95.94, 96.33, 96.65, 96.88, 97.14, 97.32, 97.48]
dist_acc = [68.68, 84.25, 89.87, 92.31, 93.81, 94.74, 95.42, 95.94, 96.36, 96.65, 96.92, 97.11, 97.35, 97.45, 97.67]

# Loss Data
base_loss = [0.5487, 0.3623, 0.2636, 0.2082, 0.1730, 0.1490, 0.1297, 0.1160, 0.1045, 0.0959, 0.0888, 0.0825, 0.0759, 0.0714, 0.0678]
dist_loss = [0.7925, 0.5433, 0.4304, 0.3744, 0.3386, 0.3169, 0.3002, 0.2874, 0.2767, 0.2694, 0.2623, 0.2578, 0.2513, 0.2484, 0.2429]
# Note: Distillation loss is higher because it sums (BCE + Cosine)

def plot_accuracy():
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, dist_acc, 'b-o', label='Ours (Distilled MobileNet)', linewidth=2)
    plt.plot(epochs, base_acc, 'r--s', label='Baseline (Standard MobileNet)', linewidth=2)
    
    plt.title('Training Convergence: Distilled vs Baseline', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(epochs)
    
    # Highlight the gap
    plt.fill_between(epochs, base_acc, dist_acc, where=(np.array(dist_acc) > np.array(base_acc)), 
                     interpolate=True, color='blue', alpha=0.1, label='Performance Gain')
    
    plt.savefig('accuracy_comparison.png', dpi=300)
    print("Saved accuracy_comparison.png")

if __name__ == "__main__":
    plot_accuracy()