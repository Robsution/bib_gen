import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json
import matplotlib.colors as mcolors

def plot_geometry_heatmap(data_dir, prefix="bib_data", gridsize=100):
    print(f"Loading data from {data_dir}...")
    parents_path = os.path.join(data_dir, f"{prefix}_parents.npy")
    daughters_path = os.path.join(data_dir, f"{prefix}_daughters.npy")

    parents = np.load(parents_path, mmap_mode='r')
    daughters = np.load(daughters_path, mmap_mode='r')

    print("Calculating true family sizes...")
    family_sizes = np.sum(daughters[:, :, 0], axis=1)

    scaler_path = os.path.join(data_dir, f"{prefix}_parent_scalers.json")
    
    with open(scaler_path, 'r') as f:
        scaler_data = json.load(f)
    data_min = np.array(scaler_data['data_min'])
    data_max = np.array(scaler_data['data_max'])
    
    parents_physical = ((parents + 1.0) / 2.0) * (data_max - data_min) + data_min

    print("Extracting spatial coordinates...")
    x = parents_physical[:, 0]
    y = parents_physical[:, 1]
    z = parents_physical[:, 2]
    
    # Calculate transverse radius
    r = np.sqrt(x**2 + y**2)

    print(f"Generating heatmaps with grid resolution {gridsize}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # -----------------------------------------
    # Plot 1: Transverse Plane (X vs Y)
    # -----------------------------------------
    # C=family_sizes and reduce_C_function=np.mean calculates the average N per physical bin.
    # mincnt=1 ensures we leave completely empty space blank (white) instead of coloring it as 0.
    hb1 = ax1.hexbin(x, y, C=family_sizes, reduce_C_function=np.mean, 
                     gridsize=gridsize, cmap='plasma', mincnt=1, 
                     edgecolors='face', linewidths=0.1, norm=mcolors.LogNorm())
    ax1.set_xlabel('x [mm]', fontsize=12)
    ax1.set_ylabel('y [mm]', fontsize=12)
    ax1.set_title('Transverse Plane: Avg Family Size (X, Y)', fontsize=14)
    ax1.set_aspect('equal') 
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    cbar1 = plt.colorbar(hb1, ax=ax1)
    cbar1.set_label('Average Daughters (N)', fontsize=12)

    # -----------------------------------------
    # Plot 2: Longitudinal Plane (Z vs R)
    # -----------------------------------------
    hb2 = ax2.hexbin(z, r, C=family_sizes, reduce_C_function=np.mean, 
                     gridsize=gridsize, cmap='plasma', mincnt=1,
                     edgecolors='face', linewidths=0.1, norm=mcolors.LogNorm())
    ax2.set_xlabel('z [mm]', fontsize=12)
    ax2.set_ylabel('r (Radius) [mm]', fontsize=12)
    ax2.set_title('Longitudinal Plane: Avg Family Size (Z, R)', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    cbar2 = plt.colorbar(hb2, ax=ax2)
    cbar2.set_label('Average Daughters (N)', fontsize=12)

    # Save and close
    output_filename = "detector_multiplicity_heatmap_log.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Heatmap saved locally as '{output_filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to your .npy files')
    # Default is 100. Use a higher number (e.g., 200) for sharper details, or lower (e.g., 50) for a smoother smear.
    parser.add_argument('--grid', type=int, default=100, help='Resolution of the heatmap bins')
    args = parser.parse_args()
    
    plot_geometry_heatmap(args.data_dir, gridsize=args.grid)