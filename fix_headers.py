import numpy as np
import os
import json
import shutil

def fix_npy_header(data_dir, prefix="bib_data"):
    print(f"--- Fixing Headers for {prefix} in {data_dir} ---")
    
    # 1. Load Scalers to calculate Feature Dimension
    json_path = os.path.join(data_dir, f"{prefix}_scalers.json")
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}. Cannot infer dimensions.")
        return

    with open(json_path, 'r') as f:
        scalers = json.load(f)
    
    # Calculate Feature Dim: 1 (Mask) + 8 (Kinematics) + N_PDGs
    n_pdgs = len(scalers["pdg_categories"])
    feature_dim = 1 + 8 + n_pdgs
    print(f"Inferred Feature Dimension: {feature_dim}")

    # 2. Fix Parents File
    parents_path = os.path.join(data_dir, f"{prefix}_parents.npy")
    fixed_parents_path = os.path.join(data_dir, f"{prefix}_parents_fixed.npy")
    
    file_size_bytes = os.path.getsize(parents_path)
    # Parents shape is (N, 3), float32 is 4 bytes
    # Size = N * 3 * 4
    n_families = file_size_bytes // (3 * 4)
    
    if file_size_bytes % 12 != 0:
        print("Warning: Parents file size is not a multiple of 12 bytes. Data might be corrupted.")
    
    print(f"Detected {n_families} families from parents file size.")
    
    # Read Raw -> Write with Header
    print("Fixing Parents file...")
    # Read raw bytes as array
    raw_parents = np.memmap(parents_path, dtype='float32', mode='r', shape=(n_families, 3))
    # Write to new file WITH header
    np.save(fixed_parents_path, raw_parents)
    del raw_parents # Close memmap
    
    # 3. Fix Daughters File
    daughters_path = os.path.join(data_dir, f"{prefix}_daughters.npy")
    fixed_daughters_path = os.path.join(data_dir, f"{prefix}_daughters_fixed.npy")
    
    daughters_size_bytes = os.path.getsize(daughters_path)
    
    # Daughters shape is (N, N_max, feature_dim)
    # Size = N * N_max * feature_dim * 4
    bytes_per_family = daughters_size_bytes // n_families
    bytes_per_row = feature_dim * 4
    n_max = bytes_per_family // bytes_per_row
    
    # Validation
    expected_size = n_families * n_max * feature_dim * 4
    if daughters_size_bytes != expected_size:
        print(f"Error: Daughters file size ({daughters_size_bytes}) does not match calculated dimensions.")
        print(f"N={n_families}, N_max={n_max}, Features={feature_dim}")
        return

    print(f"Inferred Max Daughters (N_max): {n_max}")
    print("Fixing Daughters file (this might take a moment)...")
    
    # We use open_memmap for the output to handle large files without RAM crash
    # 1. Open raw input
    raw_daughters = np.memmap(
        daughters_path, 
        dtype='float32', 
        mode='r', 
        shape=(n_families, n_max, feature_dim)
    )
    
    # 2. Create fixed output (np.lib.format.open_memmap writes the header automatically)
    fixed_daughters = np.lib.format.open_memmap(
        fixed_daughters_path,
        mode='w+',
        dtype='float32',
        shape=(n_families, n_max, feature_dim)
    )
    
    # 3. Copy data
    fixed_daughters[:] = raw_daughters[:]
    
    # Flush and close
    del raw_daughters
    del fixed_daughters
    
    print("Success! Headers fixed.")
    print(f"Renaming files to replace originals...")
    
    # Backup originals (optional, delete if you are brave)
    shutil.move(parents_path, parents_path + ".bak")
    shutil.move(daughters_path, daughters_path + ".bak")
    
    # Rename fixed to original names
    shutil.move(fixed_parents_path, parents_path)
    shutil.move(fixed_daughters_path, daughters_path)
    
    print("Done. You can now run train.py.")

if __name__ == "__main__":
    # Adjust path if your data is elsewhere
    fix_npy_header("data/processed/medium/")