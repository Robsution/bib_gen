import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import yaml
from torch.utils.data import DataLoader

# Adjust these imports to match your project structure
from src.dataset import BibDataset
from src.models import MuonGenerator

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_physics(config, checkpoint_path, num_batches=20):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"--- Running Physics Evaluation on {device} ---")

    # 1. Load Dataset
    dataset = BibDataset(config["data_dir"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    # Extract dimensions
    sample_parent, sample_features, _, _ = dataset[0]
    parent_dim = sample_parent.shape[0]
    feature_dim = sample_features.shape[1]
    max_daughters = sample_features.shape[0]

    # 2. Load Generator
    generator = MuonGenerator(
        noise_dim=config["z_dim"],
        parent_dim=parent_dim,
        max_daughters=max_daughters,
        feature_dim=feature_dim,
        hidden_dims=config.get("g_hidden_dims", [256, 512, 1024])
    ).to(device)
    
    print(f"Loading weights from {checkpoint_path}...")
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()

    real_kinematics = []
    fake_kinematics = []

    print("Generating particle showers...")
    with torch.no_grad():
        for i, (real_parent, real_features, real_mask, _) in enumerate(dataloader):
            if i >= num_batches:
                break
                
            real_parent = real_parent.to(device)
            real_features = real_features.to(device)
            real_mask = real_mask.to(device)
            
            batch_size = real_parent.size(0)
            z = torch.randn(batch_size, config["z_dim"]).to(device)
            
            # Generate fake data
            fake_features, fake_mask, _ = generator(z, real_parent)
            
            # Move to CPU numpy for filtering and plotting
            real_feat_np = real_features.cpu().numpy()
            real_mask_np = real_mask.cpu().numpy()
            fake_feat_np = fake_features.cpu().numpy()
            fake_mask_np = fake_mask.cpu().numpy()
            
            # --- CRITICAL: Mask Filtering ---
            # We must flatten the arrays and strictly extract only the "real" particles
            # For the generator, we use a hard threshold of 0.5 to decide if a particle exists
            
            # Boolean masks
            valid_real = (real_mask_np > 0.5).squeeze(-1)
            valid_fake = (fake_mask_np > 0.5).squeeze(-1)
            
            # Extract valid particles (Results in a flat 2D array: [Total Valid Particles, Features])
            real_valid_particles = real_feat_np[valid_real]
            fake_valid_particles = fake_feat_np[valid_fake]
            
            real_kinematics.append(real_valid_particles)
            fake_kinematics.append(fake_valid_particles)

    # Concatenate all batches
    all_real = np.vstack(real_kinematics)
    all_fake = np.vstack(fake_kinematics)
    
    print(f"Extracted {len(all_real)} Geant4 particles and {len(all_fake)} GAN particles.")

    # 3. Plotting
    print("Plotting histograms...")
    os.makedirs(config["plot_dir"], exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Assuming the first 4 features are your primary kinematics (e.g., pT, eta, phi, E)
    # Change these labels to match whatever order your dataset uses
    feature_names = ['Kinematic Feature 0', 'Kinematic Feature 1', 'Kinematic Feature 2', 'Kinematic Feature 3']
    
    for i in range(4):
        ax = axes[i]
        
        # Extract the specific column for both
        real_col = all_real[:, i]
        fake_col = all_fake[:, i]
        
        # Calculate dynamic bins to cover both distributions
        min_val = min(np.min(real_col), np.min(fake_col))
        max_val = max(np.max(real_col), np.max(fake_col))
        bins = np.linspace(min_val, max_val, 60)
        
        # Plot Geant4
        ax.hist(real_col, bins=bins, alpha=0.5, label='Geant4 (Real)', color='blue', density=True, histtype='stepfilled')
        ax.hist(real_col, bins=bins, color='blue', density=True, histtype='step', linewidth=1.5)
        
        # Plot GAN
        ax.hist(fake_col, bins=bins, alpha=0.5, label='GAN (Fake)', color='orange', density=True, histtype='stepfilled')
        ax.hist(fake_col, bins=bins, color='orange', density=True, histtype='step', linewidth=1.5)
        
        ax.set_title(feature_names[i], fontsize=14)
        ax.set_ylabel('Normalized Density')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plot_path = os.path.join(config["plot_dir"], "physics_evaluation.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Done! Plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to generator_epoch_X.pth')
    args = parser.parse_args()
    
    config = load_config(args.config)
    evaluate_physics(config, args.checkpoint)