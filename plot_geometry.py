import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import matplotlib.colors as colors

def plot_geometry(data_dir, prefix="bib_data"):
    print(f"Loading data from {data_dir}...")
    parents_path = os.path.join(data_dir, f"{prefix}_parents.npy")
    daughters_path = os.path.join(data_dir, f"{prefix}_daughters.npy")

    # Using mmap_mode to prevent blowing up your RAM if the dataset is massive
    parents = np.load(parents_path, mmap_mode='r')
    daughters = np.load(daughters_path, mmap_mode='r')

    print("Calculating true family sizes...")
    # The mask is at index 0 of the features. 
    # We sum across the max_daughters dimension (axis=1) to get the true N for each parent.
    family_sizes = np.sum(daughters[:, :, 0], axis=1)

    print("Extracting spatial coordinates...")
    # Assuming parents shape is (Batch, 3) -> (x, y, z)
    x = parents[:, 0]
    y = parents[:, 1]
    z = parents[:, 2]
    
    # Calculate transverse radius
    r = np.sqrt(x**2 + y**2)

    print("Generating plots...")
    # Create a wide figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # We use 'plasma' or 'viridis' colormaps because they are perceptually uniform.
    # s=2 controls the dot size, alpha=0.6 makes them slightly transparent so dense clusters blend.
    
    # -----------------------------------------
    # Plot 1: Transverse Plane (X vs Y)
    # -----------------------------------------
    # Added norm=colors.LogNorm(...)
    sc1 = ax1.scatter(x, y, c=family_sizes, cmap='plasma', s=2, alpha=0.6, 
                      norm=colors.LogNorm(vmin=max(1, family_sizes.min()), vmax=family_sizes.max()))
    
    ax1.set_xlabel('x [mm]', fontsize=12)
    ax1.set_ylabel('y [mm]', fontsize=12)
    ax1.set_xlim(-600,600)
    ax1.set_ylim(-600,600)
    ax1.set_title('Transverse Plane: Family Size vs. Decay (X, Y)', fontsize=14)
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label('Log(Number of Daughters)', fontsize=12)

    # -----------------------------------------
    # Plot 2: Longitudinal Plane (Z vs R)
    # -----------------------------------------
    # Added norm=colors.LogNorm(...)
    sc2 = ax2.scatter(z, r, c=family_sizes, cmap='plasma', s=2, alpha=0.6,
                      norm=colors.LogNorm(vmin=max(1, family_sizes.min()), vmax=family_sizes.max()))
    
    ax2.set_xlabel('z [mm]', fontsize=12)
    ax2.set_ylabel('r (Radius) [mm]', fontsize=12)
    ax2.set_xlim(-6000,0)
    ax2.set_ylim(0,600)
    ax2.set_title('Longitudinal Plane: Family Size vs. Decay (Z, R)', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label('Log(Number of Daughters)', fontsize=12)

    # Save and close
    output_filename = "detector_multiplicity_nozzle2.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Map saved locally as '{output_filename}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to your .npy files')
    args = parser.parse_args()
    
    plot_geometry(args.data_dir)