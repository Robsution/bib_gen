import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import yaml
import matplotlib.pyplot as plt

from src.dataset import BibDataset
from src.models.models_multiplicity import (
    CategoricalMultiplicity, 
    NegBinomMultiplicity, 
    MDNMultiplicity, 
    DequantizedFlowMultiplicity
)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_benchmark(history, save_dir, current_epoch):
    """Generates the presentation-ready benchmark graphs."""
    epochs = range(1, current_epoch + 2)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Architecture Bake-Off: Multiplicity Prediction (N)", fontsize=16)

    # 1. EMD Plot (The primary physical metric for stochastic counts)
    axs[0].set_title("Distributional Error (EMD)")
    for name, data in history.items():
        axs[0].plot(epochs, data["emd"], label=name, linewidth=2)
    axs[0].set_ylabel("Earth Mover's Distance (Lower is Better)")
    axs[0].set_xlabel("Epoch")
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend()

    # 2. Loss / NLL Plot (Convergence tracking)
    axs[1].set_title("Training Loss / NLL")
    for name, data in history.items():
        axs[1].plot(epochs, data["loss"], label=name, linewidth=2)
    axs[1].set_ylabel("Loss (Log Scale)")
    axs[1].set_xlabel("Epoch")
    axs[1].set_yscale('symlog') 
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, "multiplicity_benchmark.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

def calc_emd(pred_probs, true_n, max_daughters):
    """
    Calculates 1D Earth Mover's Distance (Wasserstein-1) for a batch.
    pred_probs: Tensor of shape [Batch, max_daughters + 1]
    true_n: Tensor of actual particle counts [Batch]
    """
    # Normalize probabilities over the truncated domain [0, max_daughters]
    pred_probs = pred_probs / (pred_probs.sum(dim=-1, keepdim=True) + 1e-8)
    
    # 1. Aggregate predicted probabilities across the batch
    pred_hist = pred_probs.mean(dim=0)
    
    # 2. Build the true histogram for the batch
    true_hist = torch.bincount(true_n.long(), minlength=max_daughters + 1).float() / true_n.size(0)
    
    # 3. Calculate EMD via Cumulative Distribution Functions
    pred_cdf = torch.cumsum(pred_hist, dim=0)
    true_cdf = torch.cumsum(true_hist, dim=0)
    return torch.sum(torch.abs(pred_cdf - true_cdf)).item()

def train_all_benchmarks(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Simultaneous Multiplicity Bake-Off on {device} ---")
    
    os.makedirs(config["plot_dir"], exist_ok=True)
    max_daughters = config.get("max_daughters", 50)
    
    # 1. Load Data
    print("Loading dataset...")
    dataset = BibDataset(config["data_dir"], min_n=1, max_n=max_daughters)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
    
    sample_parent, _, _, _ = dataset[0]
    parent_dim = sample_parent.shape[0]

    # 2. Initialize All Models
    models = {
        "Categorical (Softmax)": CategoricalMultiplicity(parent_dim, max_n=max_daughters).to(device),
        "Negative Binomial": NegBinomMultiplicity(parent_dim).to(device),
        "MDN (Gaussians)": MDNMultiplicity(parent_dim, num_gaussians=5).to(device),
        "Normalizing Flow": DequantizedFlowMultiplicity(parent_dim).to(device)
    }

    optimizers = {name: optim.Adam(model.parameters(), lr=1e-3) for name, model in models.items()}
    global_history = {name: {"loss": [], "emd": []} for name in models.keys()}
    
    # Common evaluation grid [0, 1, 2, ..., max_daughters]
    k_grid_1d = torch.arange(max_daughters + 1, device=device).float().unsqueeze(0) # [1, K]
    k_grid_mdn = torch.arange(max_daughters + 1, device=device).float().view(-1, 1, 1) # [K, 1, 1] for broadcasting
    
    # 3. Unified Training Loop
    print("Starting simultaneous training...")
    for epoch in range(50):
        for model in models.values():
            model.train()
            
        epoch_metrics = {name: {"loss": [], "emd": []} for name in models.keys()}
        
        for parent, _, _, scaled_n in dataloader:
            parent = parent.to(device)
            true_n = (scaled_n.to(device) * max_daughters).round().float()
            
            # --- Model 1: Categorical ---
            opt = optimizers["Categorical (Softmax)"]
            opt.zero_grad()
            logits = models["Categorical (Softmax)"](parent)
            loss_cat = nn.CrossEntropyLoss()(logits, true_n.long())
            loss_cat.backward()
            opt.step()
            
            probs_cat = torch.softmax(logits, dim=-1)
            epoch_metrics["Categorical (Softmax)"]["loss"].append(loss_cat.item())
            epoch_metrics["Categorical (Softmax)"]["emd"].append(calc_emd(probs_cat, true_n, max_daughters))

            # --- Model 2: Negative Binomial ---
            opt = optimizers["Negative Binomial"]
            opt.zero_grad()
            total_count, logits_nb = models["Negative Binomial"](parent)
            dist_nb = torch.distributions.NegativeBinomial(total_count, logits=logits_nb)
            loss_nb = -dist_nb.log_prob(true_n.unsqueeze(1)).mean()
            loss_nb.backward()
            opt.step()
            
            probs_nb = torch.exp(dist_nb.log_prob(k_grid_1d)) # Evaluate PMF across all possible N
            epoch_metrics["Negative Binomial"]["loss"].append(loss_nb.item())
            epoch_metrics["Negative Binomial"]["emd"].append(calc_emd(probs_nb, true_n, max_daughters))

            # --- Model 3: MDN ---
            opt = optimizers["MDN (Gaussians)"]
            opt.zero_grad()
            pi, mu, sigma = models["MDN (Gaussians)"](parent)
            dist_mdn = torch.distributions.Normal(mu, sigma)
            log_prob_mdn = dist_mdn.log_prob(true_n.unsqueeze(1))
            loss_mdn = -torch.logsumexp(torch.log(pi + 1e-8) + log_prob_mdn, dim=-1).mean()
            loss_mdn.backward()
            opt.step()
            
            # Evaluate mixture PDF across all possible N
            log_probs_grid = dist_mdn.log_prob(k_grid_mdn).permute(1, 0, 2) # Broadcast to [Batch, K, Components]
            probs_mdn = torch.sum(pi.unsqueeze(1) * torch.exp(log_probs_grid), dim=-1)
            epoch_metrics["MDN (Gaussians)"]["loss"].append(loss_mdn.item())
            epoch_metrics["MDN (Gaussians)"]["emd"].append(calc_emd(probs_mdn, true_n, max_daughters))

            # --- Model 4: Normalizing Flow (Dequantized) ---
            opt = optimizers["Normalizing Flow"]
            opt.zero_grad()
            mu_f, sigma_f = models["Normalizing Flow"](parent)
            noise = (torch.rand_like(true_n.unsqueeze(1)) - 0.5).to(device)
            y_dequantized = true_n.unsqueeze(1) + noise
            dist_flow = torch.distributions.Normal(mu_f, sigma_f)
            loss_flow = -dist_flow.log_prob(y_dequantized).mean()
            loss_flow.backward()
            opt.step()
            
            probs_flow = torch.exp(dist_flow.log_prob(k_grid_1d)) # Approximate P(N) via density
            epoch_metrics["Normalizing Flow"]["loss"].append(loss_flow.item())
            epoch_metrics["Normalizing Flow"]["emd"].append(calc_emd(probs_flow, true_n, max_daughters))

        # --- End of Epoch Processing ---
        print(f"\n[Epoch {epoch+1}/50] Benchmark Results:")
        print("-" * 75)
        print(f"{'Architecture':<25} | {'Loss / NLL':<15} | {'Dist. Error (EMD)':<20}")
        print("-" * 75)
        
        for name in models.keys():
            avg_loss = np.mean(epoch_metrics[name]['loss'])
            avg_emd = np.mean(epoch_metrics[name]['emd'])
            
            global_history[name]["loss"].append(avg_loss)
            global_history[name]["emd"].append(avg_emd)
            
            print(f"{name:<25} | {avg_loss:<15.4f} | {avg_emd:<20.4f}")
        print("-" * 75)
        
        plot_benchmark(global_history, config["plot_dir"], epoch)
        print(f"--> Saved updated graph to {config['plot_dir']}/multiplicity_benchmark.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_multi.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_all_benchmarks(config)