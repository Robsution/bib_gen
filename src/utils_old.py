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
            "w_dist": [],      # Wasserstein Distance estimate
            "grad_norm": [],   # Gradient Penalty Norm
            "aux_loss": [],    # Particle Count MSE
            "real_scores": [], # Average D(real) prob
            "fake_scores": []  # Average D(fake) prob
        }
        
    def update(self, d_loss, g_loss, w_dist, grad_norm, aux_loss, real_score, fake_score):
        """
        Call this at the end of every epoch.
        """
        self.history["d_loss"].append(d_loss)
        self.history["g_loss"].append(g_loss)
        self.history["w_dist"].append(w_dist)
        self.history["grad_norm"].append(grad_norm)
        self.history["aux_loss"].append(aux_loss)
        self.history["real_scores"].append(real_score)
        self.history["fake_scores"].append(fake_score)

    def plot_stats(self, epoch):
        """
        Generates a 2x2 dashboard of training health
        """
        epochs = range(len(self.history["d_loss"]))
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        plt.suptitle(f"Training Diagnostics - Epoch {epoch}", fontsize=16)

        # Plot 1: Wasserstein Distance
        axs[0, 0].plot(epochs, self.history["w_dist"], label="Est. Wasserstein Dist", color='purple')
        axs[0, 0].set_title("Wasserstein Distance (Should decrease)")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend()

        # Plot 2: Gradient Norm
        axs[0, 1].plot(epochs, self.history["grad_norm"], label="Gradient Norm", color='orange')
        axs[0, 1].axhline(y=1.0, color='r', linestyle='--', label="Target (1.0)")
        axs[0, 1].set_title("Critic Gradient Norm (Should stay near 1.0)")
        axs[0, 1].set_ylim(0, 5) 
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].legend()

        # Plot 3: Component Losses
        axs[1, 0].plot(epochs, self.history["g_loss"], label="Total G Loss", alpha=0.6)
        axs[1, 0].plot(epochs, self.history["aux_loss"], label="Aux Count Loss (MSE)", linestyle='--')
        axs[1, 0].set_title("Generator Loss Breakdown")
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)

        # Plot 4: D Probabilities (Real vs Fake)
        axs[1, 1].plot(epochs, self.history["real_scores"], label="P(Real) - Goal: ~0.5", color='g')
        axs[1, 1].plot(epochs, self.history["fake_scores"], label="P(Fake) - Goal: ~0.5", color='r')
        axs[1, 1].set_title("Discriminator Probabilities (Sigmoid)")
        axs[1, 1].set_ylim(0, 1)
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "training_dashboard.png"))
        plt.close()

    def plot_roc(self, generator, discriminator, dataloader, config, epoch, device):
        """
        Generates Real vs Fake probabilities and plots a ROC curve.
        """
        generator.eval()
        discriminator.eval()
        
        y_true = []
        y_scores = []
        
        # We only need one or two batches for a quick ROC check
        with torch.no_grad():
            for i, (real_parent, real_features, real_mask, _) in enumerate(dataloader):
                if i > 5: break 
                
                real_parent = real_parent.to(device)
                real_features = real_features.to(device)
                real_mask = real_mask.to(device)
                
                # 1. Real Data (Label = 1)
                # Apply Sigmoid to logits to get probability
                real_logits = discriminator(real_features, real_parent, real_mask)
                real_probs = torch.sigmoid(real_logits).cpu().numpy()
                
                y_true.extend([1] * len(real_probs))
                y_scores.extend(real_probs)
                
                # 2. Fake Data (Label = 0)
                z = torch.randn(real_parent.size(0), config["z_dim"]).to(device)
                fake_features, fake_mask, _ = generator(z, real_parent)
                
                fake_logits = discriminator(fake_features, real_parent, fake_mask)
                fake_probs = torch.sigmoid(fake_logits).cpu().numpy()
                
                y_true.extend([0] * len(fake_probs))
                y_scores.extend(fake_probs)

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