import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import yaml
import json
from collections import Counter
from torch.utils.data import DataLoader

from src.dataset import BibDataset
from src.models.model import MuonGenerator

# Standard dictionary mapping PDG codes to Greek/symbolic particle names
PDG_MAP = {
    22: 'γ', 11: 'e-', -11: 'e+',
    2112: 'n', 2212: 'p', -2212: 'p-bar',
    13: 'μ-', -13: 'μ+',
    111: 'π0', 211: 'π+', -211: 'π-',
    130: 'K0L', 310: 'K0S', 321: 'K+', -321: 'K-'
}

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def analyze_topologies(config, checkpoint_path, target_families=50_000):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"--- Analyzing Family Topologies ({target_families} events) ---")

    # 1. Load Data
    dataset = BibDataset(config["data_dir"], min_n=1, max_n=config.get("max_daughters", 50))
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    
    sample_parent, sample_features, _, _ = dataset[0]
    parent_dim = sample_parent.shape[0]
    feature_dim = sample_features.shape[1]
    max_daughters = sample_features.shape[0]
    kin_dim = 8
    pdg_dim = feature_dim - kin_dim

    # 2. Load Generator
    generator = MuonGenerator(
        noise_dim=config["z_dim"], parent_dim=parent_dim, max_daughters=max_daughters,
        kin_dim=kin_dim, pdg_dim=pdg_dim, hidden_dim=config.get("g_hidden_dim", 256)
    ).to(device)
    
    generator.load_state_dict(torch.load(checkpoint_path, map_location=device))
    generator.eval()

    # 3. Load PDG Decoder
    scalers_path = os.path.join(config["data_dir"], "bib_data_scalers.json")
    with open(scalers_path, 'r') as f:
        scaler_data = json.load(f)
    
    # Extract the exact array of original PDG codes
    pdg_categories = scaler_data.get('pdg_categories', [])

    real_topologies = []
    fake_topologies = []
    processed_count = 0

    print("Generating showers and classifying compositions...")
    with torch.no_grad():
        for real_parent, real_features, real_mask, real_n in dataloader:
            if processed_count >= target_families:
                break
                
            real_parent = real_parent.to(device)
            batch_size = real_parent.size(0)
            
            # Since this is the monolithic GAN for now, we just pass the internal noise
            # If you are using the Two-Stage model already, pass real_n as the target_n parameter
            z = torch.randn(batch_size, config["z_dim"]).to(device)
            
            # Use this line if using Monolithic GAN:
            # fake_features, fake_mask, _ = generator(z, real_parent)
            
            # Use this line if using Two-Stage Conditioned GAN:
            fake_features, fake_mask = generator(z, real_parent, real_n.to(device).float().unsqueeze(1))
            
            # Move to CPU for iteration
            r_feat = real_features.numpy()
            r_mask = real_mask.numpy()
            f_feat = fake_features.cpu().numpy()
            f_mask = fake_mask.cpu().numpy()
            
            # Process grouping per-family (row by row)
            for b in range(batch_size):
                # Process Geant4 Family
                r_valid = r_mask[b, :, 0] > 0.5
                if np.sum(r_valid) > 0:
                    r_pdg_idx = np.argmax(r_feat[b, r_valid, kin_dim:], axis=1)
                    r_codes = [pdg_categories[idx] for idx in r_pdg_idx]
                    r_names = [PDG_MAP.get(code, str(code)) for code in r_codes]
                    real_topologies.append(tuple(sorted(r_names)))

                # Process GAN Family
                f_valid = f_mask[b, :, 0] > 0.5
                if np.sum(f_valid) > 0:
                    f_pdg_idx = np.argmax(f_feat[b, f_valid, kin_dim:], axis=1)
                    f_codes = [pdg_categories[idx] for idx in f_pdg_idx]
                    f_names = [PDG_MAP.get(code, str(code)) for code in f_codes]
                    fake_topologies.append(tuple(sorted(f_names)))

            processed_count += batch_size

    # 4. Count and Plot
    print("Tallying topology frequencies...")
    real_counts = Counter(real_topologies)
    fake_counts = Counter(fake_topologies)
    
    # Identify the Top N most common topologies in the real dataset to anchor the plot
    top_n = 15
    most_common_real = real_counts.most_common(top_n)
    
    labels = [" + ".join(topology) for topology, _ in most_common_real]
    real_freqs = [count / len(real_topologies) for _, count in most_common_real]
    fake_freqs = [fake_counts.get(topology, 0) / len(fake_topologies) for topology, _ in most_common_real]

    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, real_freqs, width, label='Geant4 Target', color='blue', alpha=0.7)
    ax.bar(x + width/2, fake_freqs, width, label='GAN Prediction', color='orange', alpha=0.8)
    
    ax.set_ylabel('Relative Frequency in Dataset', fontsize=12)
    ax.set_title(f'Shower Composition / Family Topology (Top {top_n})', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    output_path = os.path.join(config["plot_dir"], "family_topology_bar.png")
    plt.savefig(output_path, dpi=300)
    print(f"Done! Topology plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to generator weights')
    args = parser.parse_args()
    
    config = load_config(args.config)
    analyze_topologies(config, args.checkpoint)