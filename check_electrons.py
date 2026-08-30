import numpy as np
import json
import matplotlib.pyplot as plt
from collections import Counter
import os
import argparse
from tqdm import tqdm

# Standard High-Energy Physics PDG ID mapping
PDG_MAP = {
    22: "Photon (γ)",
    11: "Electron (e-)",
    -11: "Positron (e+)",
    2112: "Neutron (n)",
    2212: "Proton (p)",
    13: "Muon (μ-)",
    -13: "Anti-muon (μ+)",
    111: "Pion (π0)",
    211: "Pion (π+)",
    -211: "Pion (π-)",
    221: "Eta (η)",
    130: "K-Short (K0_S)",
    321: "K-Plus (K+)",
    -321: "K-Minus (K-)",
}

def get_particle_name(pdg_code):
    return PDG_MAP.get(pdg_code, f"PDG:{pdg_code}")

def plot_single_daughter_compositions(data_dir, prefix="bib_data", chunk_size=5000, top_n=20):
    daughters_path = os.path.join(data_dir, f"{prefix}_daughters.npy")
    scalers_path = os.path.join(data_dir, f"{prefix}_scalers.json")
    
    if not os.path.exists(daughters_path) or not os.path.exists(scalers_path):
        print(f"Error: Could not find data or scalers in {data_dir}")
        return

    print("Loading PDG mapping categories...")
    with open(scalers_path, 'r') as f:
        scaler_info = json.load(f)
        pdg_categories = scaler_info['pdg_categories']

    print(f"Loading memory-mapped data from {daughters_path}...")
    daughters_disk = np.load(daughters_path, mmap_mode='r')
    total_families = daughters_disk.shape[0]
    
    composition_counter = Counter()
    total_1_daughter_families = 0

    print("Scanning dataset for exactly 1-daughter families...")
    for start_idx in tqdm(range(0, total_families, chunk_size)):
        end_idx = min(start_idx + chunk_size, total_families)
        chunk = daughters_disk[start_idx:end_idx]
        
        # Column 0 is the boolean mask
        masks = chunk[:, :, 0] > 0.5
        
        # Sum across the sequence length (axis 1) to get particle count per family
        daughter_counts = np.sum(masks, axis=1)
        
        # Isolate indices where exactly 1 daughter exists
        valid_indices = np.where(daughter_counts == 1)[0]
        total_1_daughter_families += len(valid_indices)
        
        if len(valid_indices) == 0:
            continue
            
        # Filter the chunk and masks down to just these families
        chunk_1d = chunk[valid_indices]
        masks_1d = masks[valid_indices]
        
        # The one-hot PDG columns start at index 9
        pdg_onehots = chunk_1d[:, :, 9:] 
        pdg_indices = np.argmax(pdg_onehots, axis=2)
        
        for i in range(len(chunk_1d)):
            # masks_1d[i] will have exactly one 'True'. We use it to index the correct particle.
            valid_idx = pdg_indices[i][masks_1d[i]][0] 
            
            # Map index -> True PDG Code -> Human Name
            pdg_code = pdg_categories[valid_idx]
            p_name = get_particle_name(pdg_code)
            composition_counter[p_name] += 1

    print(f"\nFound {total_1_daughter_families} families with exactly 1 daughter.")

    # --- Plotting ---
    most_common = composition_counter.most_common(top_n)
    
    if not most_common:
        print("No valid 1-daughter families found to plot.")
        return

    labels = [item[0] for item in most_common]
    counts = [item[1] for item in most_common]

    labels.reverse()
    counts.reverse()

    plt.figure(figsize=(12, 8))
    bars = plt.barh(labels, counts, color='tab:green', alpha=0.8, edgecolor='black')
    
    plt.title(f"Composition of 1-Daughter Showers (Total: {total_1_daughter_families:,})", fontsize=16, pad=20)
    plt.xlabel("Number of Occurrences in Dataset", fontsize=12)
    plt.ylabel("Particle Type", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Add exact count numbers to the end of the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + (max(counts)*0.01), bar.get_y() + bar.get_height()/2, 
                 f'{int(width):,}', va='center', fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(data_dir, "single_daughter_composition.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Done! Saved composition chart to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Directory with .npy and .json files")
    parser.add_argument('--prefix', type=str, default="bib_data", help="File prefix")
    parser.add_argument('--top_n', type=int, default=20, help="Number of top particles to display")
    
    args = parser.parse_args()
    plot_single_daughter_compositions(args.data_dir, prefix=args.prefix, top_n=args.top_n)