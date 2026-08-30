import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import yaml
import json
from torch.utils.data import DataLoader

from src.dataset import BibDataset
from src.models import MuonGenerator

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_million_particles(config, checkpoint_path, target_particles=1_000_000):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"--- Production Physics Validation Dashboard ({target_particles} particles) ---")

    # 1. Initialize Dataset & Dataloader to harvest real conditioning Parents
    dataset = BibDataset(config["data_dir"], min_n=2, max_n=2)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    
    sample_parent, sample_features, _, _ = dataset[0]
    parent_dim = sample_parent.shape[0]
    feature_dim = sample_features.shape[1] # Total feature space (e.g. 24)
    max_daughters = sample_features.shape[0]

    # 2. Re-instantiate and load your last stable checkpoint
    generator = MuonGenerator(
        noise_dim=config["z_dim"],
        parent_dim=parent_dim,
        max_daughters=max_daughters,
        feature_dim=feature_dim,
        hidden_dims=config.get("g_hidden_dims", [256, 512, 1024])
    ).to(device)
    
    print(f"Loading generator state from {checkpoint_path}...")
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()

    # 3. Load MinMaxScaler boundaries to un-squash data back into physical space
    # (Assuming daughters scaler JSON matches this naming convention)
    scalers_path = os.path.join(config["data_dir"], "bib_data_scalers.json")
    print(f"Loading inverse mapping coefficients from {scalers_path}...")
    with open(scalers_path, 'r') as f:
        scaler_data = json.load(f)
    
    # Extract min/max metrics (Shape matches your raw daughter attribute footprint)
    d_min = np.array(scaler_data['data_min'])
    d_max = np.array(scaler_data['data_max'])

    real_pool = []
    fake_pool = []
    
    current_real_count = 0
    current_fake_count = 0

    print("Running standalone conditional generation loop...")
    with torch.no_grad():
        for real_parent, real_features, real_mask, _ in dataloader:
            # Stop loading when both validation buckets are filled past 1e6
            if current_real_count >= target_particles and current_fake_count >= target_particles:
                break
                
            real_parent = real_parent.to(device)
            batch_size = real_parent.size(0)
            
            # Sample standard Gaussian noise vectors
            z = torch.randn(batch_size, config["z_dim"]).to(device)
            fake_features, fake_mask, _ = generator(z, real_parent)
            
            # Convert to CPU arrays for boolean mask unpadding
            r_feat = real_features.numpy()
            r_mask = real_mask.numpy()
            f_feat = fake_features.cpu().numpy()
            f_mask = fake_mask.cpu().numpy()
            
            # Apply standard mask boundaries (>0.5 means active particle)
            valid_real = (r_mask > 0.5).squeeze(-1)
            valid_fake = (f_mask > 0.5).squeeze(-1)
            
            real_particles = r_feat[valid_real]
            fake_particles = f_feat[valid_fake]
            
            if current_real_count < target_particles:
                real_pool.append(real_particles)
                current_real_count += len(real_particles)
                
            if current_fake_count < target_particles:
                fake_pool.append(fake_particles)
                current_fake_count += len(fake_particles)

    # Collate arrays into global physics matrices
    all_real_scaled = np.vstack(real_pool)[:target_particles]
    all_fake_scaled = np.vstack(fake_pool)[:target_particles]

    # ==================================================================
    # 4. UNPADDING, INVERSE SCALING, AND PDG DECODING
    # ==================================================================
    # Collate arrays into global validation matrices [Total Particles, 24]
    all_real_scaled = np.vstack(real_pool)[:target_particles]
    all_fake_scaled = np.vstack(fake_pool)[:target_particles]

    # Separate the 8 scaled kinematics from the 16 Gumbel-Softmax PDG columns
    real_kin_scaled = all_real_scaled[:, :8]
    fake_kin_scaled = all_fake_scaled[:, :8]

    # THE CORRECTED FIX: Shift from [-1, 1] to [0, 1] before multiplying by the range
    real_kin = ((real_kin_scaled + 1.0) / 2.0) * (d_max - d_min) + d_min
    fake_kin = ((fake_kin_scaled + 1.0) / 2.0) * (d_max - d_min) + d_min

    # Collapse the 16 one-hot columns back into a single integer ID column [5000, 1]
    real_pdg = np.argmax(all_real_scaled[:, 8:], axis=1).reshape(-1, 1)
    fake_pdg = np.argmax(all_fake_scaled[:, 8:], axis=1).reshape(-1, 1)

    # Recombine into a clean, physical matrix of 9 total columns
    all_real = np.hstack([real_kin, real_pdg])
    all_fake = np.hstack([fake_kin, fake_pdg])

    print(f"Successfully processed {all_real.shape[0]} physical particles.")

    # ==================================================================
    # 5. BUILD 3x3 PHYSICS DIAGNOSTICS DASHBOARD
    # ==================================================================
    print("Plotting publication diagnostics...")
    os.makedirs(config["plot_dir"], exist_ok=True)
    
    # Expanded to a 3x3 grid to beautifully fit all 8 kinematics + 1 PDG plot
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    axes = axes.flatten()

    # Verify this matches the exact kinematic order inside your dataset features!
    labels = [
        'x [mm]', 'y [mm]', 'z [mm]', 
        'px [GeV]', 'py [GeV]', 'pz [GeV]', 
        'pT [GeV]', 'phi [rad]', 'Decoded PDG Index'
    ]

    # Loop through all 9 physical parameters
    for i in range(9):
        ax = axes[i]
        real_axis = all_real[:, i]
        fake_axis = all_fake[:, i]
        
        min_v = min(np.min(real_axis), np.min(fake_axis))
        max_v = max(np.max(real_axis), np.max(fake_axis))
        if min_v == max_v:
            max_v += 1.0
            
        bins = np.linspace(min_v, max_v, 80)
        
        ax.hist(real_axis, bins=bins, alpha=0.4, label='FLUKA Target', color='blue', density=True)
        ax.hist(real_axis, bins=bins, color='blue', histtype='step', linewidth=1.2, density=True)
        
        ax.hist(fake_axis, bins=bins, alpha=0.4, label='WGAN-GP', color='orange', density=True)
        ax.hist(fake_axis, bins=bins, color='orange', histtype='step', linewidth=1.2, density=True)
        
        ax.set_title(f'{labels[i]}', fontsize=13)
        ax.set_ylabel('Normalized Density')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)

    # Hide the 9th empty plot frame if you only end up using 8 columns
    # ax.set_visible(False) 

    plt.tight_layout()
    output_img = os.path.join(config["plot_dir"], "million_particle_physics_check.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Success! Physics dashboard exported safely to: {output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to generator weights')
    args = parser.parse_args()
    
    config = load_config(args.config)
    generate_million_particles(config, args.checkpoint, target_particles=5000) 