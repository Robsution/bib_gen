import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from sklearn.metrics import roc_curve, auc

class TrainingMonitor:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.history = {
            "d_loss": [],
            "g_loss": [],
            "w_dist": [],      # Pseudo W-Distance: D(real) - D(fake)
            "real_scores": [], # Raw D(real) logits (Target: >= +1.0)
            "fake_scores": []  # Raw D(fake) logits (Target: <= -1.0)
        }
        
    def update(self, d_loss, g_loss, w_dist, real_score, fake_score):
        """
        Call this at the end of every epoch.
        """
        self.history["d_loss"].append(d_loss)
        self.history["g_loss"].append(g_loss)
        self.history["w_dist"].append(w_dist)
        self.history["real_scores"].append(real_score)
        self.history["fake_scores"].append(fake_score)

    def plot_stats(self, epoch):
        """
        Generates a 2x2 dashboard of SN-GAN Hinge training health
        """
        epochs = range(len(self.history["d_loss"]))
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        plt.suptitle(f"SN-GAN Hinge Diagnostics - Epoch {epoch}", fontsize=16)

        # Plot 1: Adversarial Losses
        axs[0, 0].plot(epochs, self.history["d_loss"], label="D Loss (Hinge)", color='tab:red')
        axs[0, 0].plot(epochs, self.history["g_loss"], label="G Loss", color='tab:green')
        axs[0, 0].set_title("Network Losses")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend()

        # Plot 2: Pseudo Wasserstein Distance (Margin)
        axs[0, 1].plot(epochs, self.history["w_dist"], label="Score Margin", color='purple')
        axs[0, 1].set_title("Pseudo W-Distance: D(real) - D(fake)")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].legend()

        # Plot 3: Raw Discriminator Scores (The Hinge Targets)
        # This is the most important plot for Hinge GANs!
        axs[1, 0].plot(epochs, self.history["real_scores"], label="D(Real) mean", color='blue')
        axs[1, 0].plot(epochs, self.history["fake_scores"], label="D(Fake) mean", color='orange')
        axs[1, 0].axhline(y=1.0, color='blue', linestyle='--', alpha=0.5, label="Real Hinge Target (+1)")
        axs[1, 0].axhline(y=-1.0, color='orange', linestyle='--', alpha=0.5, label="Fake Hinge Target (-1)")
        axs[1, 0].set_title("Raw D Scores (Should push past targets)")
        axs[1, 0].set_xlabel("Epoch")
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].legend()

        # Plot 4: Generator Loss Isolated (To spot slow decay)
        axs[1, 1].plot(epochs, self.history["g_loss"], label="G Loss", color='tab:green')
        axs[1, 1].set_title("Generator Loss (Isolated)")
        axs[1, 1].set_xlabel("Epoch")
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "training_dashboard.png"))
        plt.close()

    def plot_roc(self, generator, discriminator, dataloader, config, epoch, device):
        """
        Generates ROC curve based on RAW logits.
        """
        generator.eval()
        discriminator.eval()
        
        y_true = []
        y_scores = []
        
        with torch.no_grad():
            for i, (real_parent, real_features, real_mask, _) in enumerate(dataloader):
                if i > 5: break 
                
                real_parent = real_parent.to(device)
                real_features = real_features.to(device)
                real_mask = real_mask.to(device)
                
                # 1. Real Data (Label = 1)
                real_logits = discriminator(real_features, real_parent, real_mask)
                # Sigmoid removed - ROC works identically on rank-ordered raw logits
                y_true.extend([1] * len(real_logits))
                y_scores.extend(real_logits.cpu().numpy().flatten())
                
                # 2. Fake Data (Label = 0)
                z = torch.randn(real_parent.size(0), config["z_dim"]).to(device)
                fake_features, fake_mask, _ = generator(z, real_parent)
                
                fake_logits = discriminator(fake_features, real_parent, fake_mask)
                y_true.extend([0] * len(fake_logits))
                y_scores.extend(fake_logits.cpu().numpy().flatten())

        # Calculate ROC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Discriminator ROC - Epoch {epoch}')
        plt.legend(loc="lower right")
        
        save_path = os.path.join(self.save_dir, f"roc_epoch_{epoch}.png")
        plt.savefig(save_path)
        plt.close()
        
        # Return models to train mode
        generator.train()
        discriminator.train()