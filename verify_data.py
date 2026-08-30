import numpy as np
import os
import argparse
from tqdm import tqdm

def verify_data(data_dir, prefix="bib_data", chunk_size=10000):
    parents_path = os.path.join(data_dir, f"{prefix}_parents.npy")
    daughters_path = os.path.join(data_dir, f"{prefix}_daughters.npy")
    
    if not os.path.exists(parents_path) or not os.path.exists(daughters_path):
        print(f"Error: Could not find {parents_path} or {daughters_path}")
        return

    print(f"Loading memory maps from {data_dir}...")
    parents_mm = np.load(parents_path, mmap_mode='r')
    daughters_mm = np.load(daughters_path, mmap_mode='r')
    
    total_samples = parents_mm.shape[0]
    print(f"Total families to verify: {total_samples}")

    nan_parents = 0
    nan_daughters = 0
    empty_families = 0
    zero_kinematic_particles = 0

    for start_idx in tqdm(range(0, total_samples, chunk_size)):
        end_idx = min(start_idx + chunk_size, total_samples)
        
        p_chunk = parents_mm[start_idx:end_idx]
        d_chunk = daughters_mm[start_idx:end_idx]
        
        # 1. NaN Checks
        if np.isnan(p_chunk).any():
            nan_parents += np.isnan(p_chunk).any(axis=1).sum()
            
        if np.isnan(d_chunk).any():
            # Sum over dimensions 1 and 2 to find how many families have NaNs
            nan_daughters += np.isnan(d_chunk).any(axis=(1, 2)).sum()
            
        # 2. Empty Family Check (No active masks)
        masks = d_chunk[:, :, 0] > 0.5
        active_counts = np.sum(masks, axis=1)
        empty_families += np.sum(active_counts == 0)
        
        # 3. Valid Mask but Zero Kinematics (Dead particles)
        # Assuming kinematics are columns 1 through 8
        kinematics = d_chunk[:, :, 1:9]
        kin_sums = np.sum(np.abs(kinematics), axis=2)
        
        # Particle is "valid" but has absolute zero momentum/position
        dead_particles = (masks) & (kin_sums == 0)
        zero_kinematic_particles += np.sum(dead_particles)

    print("\n--- Verification Report ---")
    print(f"Families with NaN in Parents:   {nan_parents}")
    print(f"Families with NaN in Daughters: {nan_daughters}")
    print(f"Completely Empty Families:      {empty_families}")
    print(f"Active Particles with 0 Kinem.: {zero_kinematic_particles}")
    
    if (nan_parents + nan_daughters + empty_families + zero_kinematic_particles) == 0:
        print("\nDataset is clean.")
    else:
        print("\nWARNING: Dataset contains corruption. Check the lines matching the errors above.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--prefix', type=str, default="bib_data")
    args = parser.parse_args()
    
    verify_data(args.data_dir, args.prefix)