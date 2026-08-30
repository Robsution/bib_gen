import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import yaml
import json
import glob
import re
from torch.utils.data import DataLoader

from src.dataset import BibDataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_validation_particles(config, args):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"--- Production Physics Validation Dashboard ({args.target_particles} particles) ---")

    # Resolve Generator Checkpoint automatically if not provided
    if args.checkpoint:
        gen_ckpt = args.checkpoint
    else:
        save_dir = config.get("save_dir", ".")
        files = glob.glob(os.path.join(save_dir, "generator_epoch_*.pth"))
        if not files:
            raise ValueError(f"No generator checkpoints found in '{save_dir}'. Please specify --checkpoint.")
        
        # Extract integer epoch numbers to strictly find the highest epoch
        def extract_epoch(filepath):
            match = re.search(r'_epoch_(\d+)\.pth', filepath)
            return int(match.group(1)) if match else -1
            
        gen_ckpt = max(files, key=extract_epoch)
        print(f"Auto-detected latest generator checkpoint: {gen_ckpt}")

    # 1. Initialize Dataset & Dataloader
    dataset = BibDataset(
        config["data_dir"], 
        min_n=config.get("min_daughters", 1), 
        max_n=config.get("max_daughters", 50)
    )
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    
    sample_parent, sample_features, _, _ = dataset[0]
    parent_dim = sample_parent.shape[0]
    feature_dim = sample_features.shape[1] 
    max_daughters = config.get("max_daughters", 50)
    kin_dim = 8
    pdg_dim = feature_dim - kin_dim

    # 2. Dynamic Model Loading
    if args.model_type == 'mono':
        print("Loading Monolithic GAN...")
        from src.models.generator import MuonGenerator
        generator = MuonGenerator(
            noise_dim=config["z_dim"], parent_dim=parent_dim, max_daughters=max_daughters,
            kin_dim=kin_dim, pdg_dim=pdg_dim, hidden_dim=config.get("g_hidden_dim", 256)
        ).to(device)
    else:
        print("Loading Two-Stage Cascaded GAN...")
        from src.models.model import MuonGenerator, CategoricalOracle
        
        # Resolve Oracle Checkpoint automatically if not provided
        if args.oracle_checkpoint:
            oracle_ckpt = args.oracle_checkpoint
        else:
            save_dir = config.get("save_dir", ".")
            oracle_ckpt = os.path.join(save_dir, "oracle_weights.pth")
            if not os.path.exists(oracle_ckpt):
                raise ValueError(f"Oracle weights not found at '{oracle_ckpt}'. Please specify --oracle_checkpoint.")
            print(f"Auto-detected oracle checkpoint: {oracle_ckpt}")
            
        oracle = CategoricalOracle(parent_dim=parent_dim, max_daughters=max_daughters, hidden_dim=128).to(device)
        print(f"Loading Oracle state from {oracle_ckpt}...")
        oracle.load_state_dict(torch.load(oracle_ckpt, map_location=device))
        oracle.eval()
        
        generator = MuonGenerator(
            noise_dim=config["z_dim"], parent_dim=parent_dim, max_daughters=max_daughters,
            kin_dim=kin_dim, pdg_dim=pdg_dim, hidden_dim=config.get("g_hidden_dim", 256)
        ).to(device)
    
    print(f"Loading Generator state from {gen_ckpt}...")
    generator.load_state_dict(torch.load(gen_ckpt, map_location=device))
    generator.eval()

    # 3. Load MinMaxScaler boundaries
    scalers_path = os.path.join(config["data_dir"], "bib_data_scalers.json")
    with open(scalers_path, 'r') as f:
        scaler_data = json.load(f)
    
    d_min = np.array(scaler_data['data_min'])
    d_max = np.array(scaler_data['data_max'])

    real_pool, fake_pool = [], []
    current_real_count, current_fake_count = 0, 0

    print(f"Generating up to {args.target_particles} particles...")
    with torch.no_grad():
        for real_parent, real_features, real_mask, _ in dataloader:
            if current_real_count >= args.target_particles and current_fake_count >= args.target_particles:
                break
                
            real_parent = real_parent.to(device)
            batch_size = real_parent.size(0)
            z = torch.randn(batch_size, config["z_dim"]).to(device)
            
            # Forward Pass switching
            if args.model_type == 'mono':
                fake_features, fake_mask, _ = generator(z, real_parent)
            else:
                oracle_logits = oracle(real_parent)
                oracle_probs = torch.softmax(oracle_logits, dim=-1)
                target_n_int = torch.multinomial(oracle_probs, num_samples=1).float()
                target_n = target_n_int / max_daughters
                fake_features, fake_mask = generator(z, real_parent, target_n)
            
            # Mask Filtering
            r_feat = real_features.numpy()
            r_mask = real_mask.numpy()
            f_feat = fake_features.cpu().numpy()
            f_mask = fake_mask.cpu().numpy()
            
            valid_real = (r_mask > 0.5).squeeze(-1)
            valid_fake = (f_mask > 0.5).squeeze(-1)
            
            real_particles = r_feat[valid_real]
            fake_particles = f_feat[valid_fake]
            
            if current_real_count < args.target_particles:
                real_pool.append(real_particles)
                current_real_count += len(real_particles)
                
            if current_fake_count < args.target_particles:
                fake_pool.append(fake_particles)
                current_fake_count += len(fake_particles)

    # 4. UNPADDING, INVERSE SCALING, AND PDG DECODING
    all_real_scaled = np.vstack(real_pool)[:args.target_particles]
    all_fake_scaled = np.vstack(fake_pool)[:args.target_particles]

    real_kin_scaled = all_real_scaled[:, :8]
    fake_kin_scaled = all_fake_scaled[:, :8]

    # Shift from [-1, 1] to [0, 1] before multiplying by the physical range
    real_kin = ((real_kin_scaled + 1.0) / 2.0) * (d_max - d_min) + d_min
    fake_kin = ((fake_kin_scaled + 1.0) / 2.0) * (d_max - d_min) + d_min

    # Extract Categorical PDG
    real_pdg = np.argmax(all_real_scaled[:, 8:], axis=1).reshape(-1, 1)
    fake_pdg = np.argmax(all_fake_scaled[:, 8:], axis=1).reshape(-1, 1)

    all_real = np.hstack([real_kin, real_pdg])
    all_fake = np.hstack([fake_kin, fake_pdg])

    print(f"Successfully recovered {all_real.shape[0]} physical particles.")

    # 5. BUILD 3x3 DASHBOARD
    print("Plotting diagnostics...")
    os.makedirs(config["plot_dir"], exist_ok=True)
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 16))
    axes = axes.flatten()

    labels = [
        'x [mm]', 'y [mm]', 'z [mm]', 
        'px [GeV]', 'py [GeV]', 'pz [GeV]', 
        'Energy [GeV] / t [ns]', 'phi [rad] / unknown', 'Decoded PDG Index' # Adjusted labels to standard physics sets
    ]

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
        
        ax.hist(fake_axis, bins=bins, alpha=0.4, label=f'{args.model_type.upper()} GAN', color='orange', density=True)
        ax.hist(fake_axis, bins=bins, color='orange', histtype='step', linewidth=1.2, density=True)
        
        ax.set_title(f'{labels[i]}', fontsize=13)
        ax.set_ylabel('Normalized Density')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    output_img = os.path.join(config["plot_dir"], f"physics_check_{args.model_type}.png")
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Dashboard exported safely to: {output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--model_type', type=str, choices=['mono', 'twostage'], required=True, help="Which architecture to evaluate")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to generator weights (overrides config)')
    parser.add_argument('--oracle_checkpoint', type=str, default=None, help='Path to Oracle weights (overrides config)')
    parser.add_argument('--target_particles', type=int, default=1000000, help='Number of valid particles to aggregate')
    args = parser.parse_args()
    
    config = load_config(args.config)
    generate_validation_particles(config, args)