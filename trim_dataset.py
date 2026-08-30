import numpy as np
import os
import argparse
from tqdm import tqdm

def trim_dataset(data_dir, output_dir, prefix="bib_data", max_n=50, chunk_size=5000):
    parents_path = os.path.join(data_dir, f"{prefix}_parents.npy")
    daughters_path = os.path.join(data_dir, f"{prefix}_daughters.npy")
    
    os.makedirs(output_dir, exist_ok=True)
    out_parents_path = os.path.join(output_dir, f"{prefix}_parents_trimmed.npy")
    out_daughters_path = os.path.join(output_dir, f"{prefix}_daughters_trimmed.npy")

    print(f"Loading memory maps from {data_dir}...")
    # mmap_mode='r' prevents loading the massive file into RAM
    parents_disk = np.load(parents_path, mmap_mode='r')
    daughters_disk = np.load(daughters_path, mmap_mode='r')
    
    total_samples = parents_disk.shape[0]
    feature_dim = daughters_disk.shape[2]
    
    print(f"Original Dataset Size: {total_samples} families")
    print(f"Original Matrix Shape: {daughters_disk.shape}")
    
    trimmed_parents = []
    trimmed_daughters = []
    
    print(f"Filtering families with <= {max_n} daughters (Processing in chunks to save RAM)...")
    
    # Process in chunks to prevent WSL from OOM crashing
    for start_idx in tqdm(range(0, total_samples, chunk_size)):
        end_idx = min(start_idx + chunk_size, total_samples)
        
        # Load just this specific chunk into active RAM
        chunk_parents = parents_disk[start_idx:end_idx]
        chunk_daughters = daughters_disk[start_idx:end_idx]
        
        # The mask is column 0. Count real particles.
        # We use > 0.5 to safely handle floating point masks
        chunk_masks = chunk_daughters[:, :, 0] > 0.5
        true_counts = np.sum(chunk_masks, axis=1)
        
        # Find indices in this chunk that have between 1 and max_n daughters
        valid_indices = np.where((true_counts > 0) & (true_counts <= max_n))[0]
        
        if len(valid_indices) > 0:
            # Extract the valid parents
            valid_parents = chunk_parents[valid_indices]
            
            # Extract the valid daughters AND strictly slice the column size down to max_n
            valid_daughters = chunk_daughters[valid_indices, :max_n, :]
            
            trimmed_parents.append(valid_parents)
            trimmed_daughters.append(valid_daughters)

    if not trimmed_parents:
        print(f"Error: No families found with {max_n} or fewer daughters.")
        return

    print("Concatenating trimmed chunks...")
    final_parents = np.vstack(trimmed_parents)
    final_daughters = np.vstack(trimmed_daughters)
    
    print(f"New Dataset Size: {final_parents.shape[0]} families (Kept {(final_parents.shape[0]/total_samples)*100:.1f}%)")
    print(f"New Matrix Shape: {final_daughters.shape}")
    
    print(f"Saving to {output_dir}...")
    np.save(out_parents_path, final_parents)
    np.save(out_daughters_path, final_daughters)
    
    # If you have scaler JSONs, you should manually copy them over to the new directory 
    # since the MinMax boundaries for the features haven't fundamentally changed.
    print("Done! Note: Remember to copy your scaler JSON files to the new directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to your huge .npy files')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save the trimmed files')
    parser.add_argument('--max_daughters', type=int, default=50, help='Maximum daughters to keep')
    args = parser.parse_args()
    
    trim_dataset(args.data_dir, args.output_dir, max_n=args.max_daughters)