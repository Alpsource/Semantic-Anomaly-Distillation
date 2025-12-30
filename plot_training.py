import matplotlib.pyplot as plt
import numpy as np

# Data extracted from training logs!!!
epochs = range(1, 16)

# Accuracy Data
base_acc = [
    70.72, 85.92, 91.12, 93.31, 94.63, 
    95.50, 96.04, 96.46, 96.88, 97.12, 
    97.27, 97.55, 97.71, 97.83, 97.97
]
dist_acc = [68.68, 84.25, 89.87, 92.31, 93.81, 94.74, 95.42, 95.94, 96.36, 96.65, 96.92, 97.11, 97.35, 97.45, 97.67]

# Loss Data
base_loss = [
    0.5497, 0.3244, 0.2176, 0.1664, 0.1374, 
    0.1163, 0.1026, 0.0924, 0.0827, 0.0769, 
    0.0720, 0.0655, 0.0631, 0.0592, 0.0557
]
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