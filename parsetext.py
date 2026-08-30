import re
import matplotlib.pyplot as plt
import os

def parse_and_plot(log_file="training_log.txt", save_path="recovered_dashboard.png"):
    if not os.path.exists(log_file):
        print(f"Error: Could not find {log_file}. Please save your text to this file.")
        return

    # Data arrays
    epochs = []
    d_losses = []
    g_losses = []
    w_dists = []
    aux_mses = []

    # Regex pattern to strictly match and extract the numbers from your log format
    # Example line: [Epoch 1/3000] D Loss: 0.6330 | G Loss: 0.3191 | W Dist: 0.0132 | Aux MSE: 0.1190
    pattern = re.compile(
        r"\[Epoch (\d+)/\d+\] D Loss: ([-\.\d]+) \| G Loss: ([-\.\d]+) \| W Dist: ([-\.\d]+) \| Aux MSE: ([-\.\d]+)"
    )

    print("Parsing log file...")
    with open(log_file, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                d_losses.append(float(match.group(2)))
                g_losses.append(float(match.group(3)))
                w_dists.append(float(match.group(4)))
                aux_mses.append(float(match.group(5)))

    if not epochs:
        print("No valid data found in the log file. Check your file formatting.")
        return

    print(f"Successfully extracted {len(epochs)} epochs of data.")
    print("Generating dashboard...")

    # Recreate the Dashboard layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Recovered WGAN Training Dashboard', fontsize=18)

    # 1. Discriminator Loss
    axes[0, 0].plot(epochs, d_losses, color='tab:red', linewidth=1.5)
    axes[0, 0].set_title('Discriminator (Critic) Loss', fontsize=14)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    # 2. Generator Loss
    axes[0, 1].plot(epochs, g_losses, color='tab:green', linewidth=1.5)
    axes[0, 1].set_title('Generator Loss', fontsize=14)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    # 3. Wasserstein Distance
    axes[1, 0].plot(epochs, w_dists, color='tab:blue', linewidth=1.5)
    axes[1, 0].set_title('Wasserstein Distance', fontsize=14)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('W-Distance')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.8) # Baseline reference
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    # 4. Aux MSE Loss
    axes[1, 1].plot(epochs, aux_mses, color='tab:purple', linewidth=1.5)
    axes[1, 1].set_title('Auxiliary Multiplicity Loss (MSE)', fontsize=14)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE Loss')
    # Set a log scale on the Y-axis if you have massive spikes, otherwise keep linear
    axes[1, 1].set_yscale('log') 
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.savefig(save_path, dpi=300)
    print(f"Done! Plot saved to {save_path}")

if __name__ == "__main__":
    parse_and_plot()