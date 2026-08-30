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

def plot_family_compositions(data_dir, prefix="bib_data", chunk_size=5000, top_n=20):
    daughters_path = os.path.join(data_dir, f"{prefix}_daughters.npy") # Or _daughters_trimmed.npy
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

    print("Scanning dataset for family compositions...")
    for start_idx in tqdm(range(0, total_families, chunk_size)):
        end_idx = min(start_idx + chunk_size, total_families)
        chunk = daughters_disk[start_idx:end_idx]
        
        # Extract features
        masks = chunk[:, :, 0] > 0.5
        # The one-hot PDG columns start at index 9 and go to 24 (assuming 16 categories)
        pdg_onehots = chunk[:, :, 9:] 
        pdg_indices = np.argmax(pdg_onehots, axis=2)
        
        for i in range(len(chunk)):
            valid_mask = masks[i]
            if not np.any(valid_mask):
                continue # Skip completely empty families
                
            # Get valid particle indices for this specific family
            family_indices = pdg_indices[i][valid_mask]
            
            # Map index -> True PDG Code -> Human Name
            family_pdgs = [pdg_categories[idx] for idx in family_indices]
            family_names = [get_particle_name(pdg) for pdg in family_pdgs]
            
            # Sort the names alphabetically so (e-, γ) is counted the same as (γ, e-)
            family_names.sort()
            
            # Create a string representation (e.g., "Electron (e-) + Photon (γ)")
            composition_str = " + ".join(family_names)
            composition_counter[composition_str] += 1

    # --- Plotting ---
    print("\nGenerating composition chart...")
    most_common = composition_counter.most_common(top_n)
    
    if not most_common:
        print("No valid families found to plot.")
        return

    labels = [item[0] for item in most_common]
    counts = [item[1] for item in most_common]

    # Reverse lists so the largest bar is at the top of the horizontal chart
    labels.reverse()
    counts.reverse()

    plt.figure(figsize=(14, 10))
    bars = plt.barh(labels, counts, color='tab:blue', alpha=0.8, edgecolor='black')
    
    plt.title(f"Top {top_n} Most Common Beam-Induced Background Showers", fontsize=16, pad=20)
    plt.xlabel("Number of Occurrences in Dataset", fontsize=12)
    plt.ylabel("Shower Composition", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Add exact count numbers to the end of the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + (max(counts)*0.01), bar.get_y() + bar.get_height()/2, 
                 f'{int(width):,}', va='center', fontsize=10)

    # Make room for long labels
    plt.tight_layout()
    
    plot_path = os.path.join(data_dir, "family_compositions.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Done! Saved composition chart to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Point this to whichever directory holds your current targets (e.g. data/processed/large)
    parser.add_argument('--data_dir', type=str, required=True, help="Directory with .npy and .json files")
    parser.add_argument('--prefix', type=str, default="bib_data", help="File prefix (e.g., bib_data or bib_data_trimmed)")
    parser.add_argument('--top_n', type=int, default=20, help="Number of top compositions to display")
    
    args = parser.parse_args()
    plot_family_compositions(args.data_dir, prefix=args.prefix, top_n=args.top_n)