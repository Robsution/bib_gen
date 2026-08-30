import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# --- HELPER: Raw File Generator to bypass CSV limits ---
def raw_line_chunk_generator(filepath, chunk_size=5000):
    """
    Yields chunks of raw lines from the file.
    Bypasses csv module limits entirely.
    """
    with open(filepath, 'r') as f:
        batch = []
        for line in f:
            if line.strip(): # Skip empty lines
                    batch.append(line.strip())
            
            if len(batch) >= chunk_size:
                yield batch
                batch = []
        if batch:
            yield batch

def process_bib_data_robust(csv_path, output_prefix="bib_data", folder="data/processed/", chunk_size=5000):
    # Ensure output folder exists
    os.makedirs(folder, exist_ok=True)
    print(f"--- Processing {csv_path} ---")
    print("Method: Raw Line Reading -> Header-aware Memmap")



    print("Phase 1: Scanning file for metadata and statistics...")
    
    max_cols = 0
    total_rows = 0
    unique_pdgs = set()
    
    # Kinematic Min/Max trackers (3 parent features x, y, z) 
    # (8 features: E, x, y, z, px, py, pz, t)
    parent_global_min = np.full(3, np.inf)
    parent_global_max = np.full(3, -np.inf)
    global_min = np.full(8, np.inf)
    global_max = np.full(8, -np.inf)

    # Iterate using our custom generator
    for raw_lines in raw_line_chunk_generator(csv_path, chunk_size):
        # 1. Create Series
        s = pd.Series(raw_lines)
        
        # 2. Split (expand=True creates a DataFrame)
        temp_df = s.str.split(',', expand=True)
        
        # Update metadata
        current_cols = temp_df.shape[1]
        max_cols = max(max_cols, current_cols)
        total_rows += len(temp_df)
        
        # 3. Convert to numeric (coercing errors to NaN)
        vals = temp_df.apply(pd.to_numeric, errors='coerce').values
        
        # 4. Extract Daughters (Cols 3 onwards)
        if vals.shape[1] > 3:
            parent_vals = vals[:, 0:3]
            daughter_vals = vals[:, 3:]
            n_daughters_in_chunk = daughter_vals.shape[1] // 11
            
            for d_idx in range(n_daughters_in_chunk):
                start = d_idx * 11
                end = start + 11
                if end > daughter_vals.shape[1]: break
                
                particle_chunk = daughter_vals[:, start:end]
                valid_mask = ~np.isnan(particle_chunk[:, 0])
                valid_particles = particle_chunk[valid_mask]
                
                if len(valid_particles) > 0:
                    unique_pdgs.update(valid_particles[:, 0].astype(int))
                    kinematics = valid_particles[:, 1:9]
                    
                    parent_batch_min = np.min(parent_vals, axis = 0)
                    parent_batch_max = np.max(parent_vals, axis = 0)
                    parent_global_min = np.minimum(parent_global_min, parent_batch_min)
                    parent_global_max = np.maximum(parent_global_max, parent_batch_max)
                    
                    batch_min = np.min(kinematics, axis=0)
                    batch_max = np.max(kinematics, axis=0)
                    global_min = np.minimum(global_min, batch_min)
                    global_max = np.maximum(global_max, batch_max)

        print(f"  Scanned {total_rows} rows...", end='\r')

    print(f"\nScan complete. Found {total_rows} families. Max columns: {max_cols}")
    
    # Calculate N_max
    N_max = (max_cols - 3) // 11
    print(f"Max daughters (N_max): {N_max}")
    print(f"Unique PDGs found: {sorted(list(unique_pdgs))}")



    print("Phase 2: Initializing Scalers and Output Files...")
    
    # Fit PDG Encoder
    sorted_pdgs = sorted(list(unique_pdgs))
    pdg_encoder = OneHotEncoder(categories=[sorted_pdgs], sparse_output=False)
    pdg_encoder.fit(np.array(sorted_pdgs).reshape(-1, 1))
    num_pdg_features = len(sorted_pdgs)
    
    # Fit MinMax Scaler for daughters and parents
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit([global_min, global_max])
    
    parent_scaler = MinMaxScaler(feature_range=(-1, 1))
    parent_scaler.fit([parent_global_min, parent_global_max])
    
    # Create Header-Aware Memmaps
    parent_shape = (total_rows, 3)
    final_feature_dim = 1 + 8 + num_pdg_features 
    daughter_shape = (total_rows, N_max, final_feature_dim)
    
    # np.lib.format.open_memmap writes the header correctly so np.load works later
    parents_mm = np.lib.format.open_memmap(
        f"{folder}{output_prefix}_parents.npy", 
        mode='w+', 
        dtype='float32', 
        shape=parent_shape
    )
    
    daughters_mm = np.lib.format.open_memmap(
        f"{folder}{output_prefix}_daughters.npy", 
        mode='w+', 
        dtype='float32', 
        shape=daughter_shape
    )
    
    print(f"Initialized .npy files. Daughters shape: {daughter_shape}")




    print("Phase 3: Processing and Writing data...")
    
    processed_count = 0
    
    for raw_lines in raw_line_chunk_generator(csv_path, chunk_size):
        
        # 1. Parse Chunk
        s = pd.Series(raw_lines)
        temp_df = s.str.split(',', expand=True)
        
        # Pad chunk to match max_cols
        if temp_df.shape[1] < max_cols:
            padding = pd.DataFrame(np.nan, index=temp_df.index, columns=range(temp_df.shape[1], max_cols))
            temp_df = pd.concat([temp_df, padding], axis=1)
            
        vals = temp_df.apply(pd.to_numeric, errors='coerce').values
        
        current_batch_size = len(vals)
        start_idx = processed_count
        end_idx = processed_count + current_batch_size
        
        # 2. Write Parents
        parents_mm[start_idx:end_idx] = parent_scaler.transform(vals[:, 0:3])
        
        # 3. Process Daughters
        batch_tensor = np.zeros((current_batch_size, N_max, final_feature_dim), dtype='float32')
        daughter_vals = vals[:, 3:]
        
        for i in range(N_max):
            col_start = i * 11
            col_end = col_start + 11
            if col_end > daughter_vals.shape[1]: break
            
            raw_particle = daughter_vals[:, col_start:col_end]
            
            # Mask
            is_valid = ~np.isnan(raw_particle[:, 0])
            valid_indices = np.where(is_valid)[0]
            
            if len(valid_indices) > 0:
                valid_data = raw_particle[valid_indices]
                
                # Encode & Scale
                pdgs_col = valid_data[:, 0].reshape(-1, 1)
                pdgs_onehot = pdg_encoder.transform(pdgs_col)
                
                kinematics = valid_data[:, 1:9]
                kinematics_scaled = scaler.transform(kinematics)
                
                # Assemble
                mask_vec = np.ones((len(valid_indices), 1))
                features = np.hstack([mask_vec, kinematics_scaled, pdgs_onehot])
                
                batch_tensor[valid_indices, i, :] = features
        
        # 4. Write batch
        daughters_mm[start_idx:end_idx] = batch_tensor
        
        # Flush
        if processed_count % (chunk_size * 5) == 0:
            parents_mm.flush()
            daughters_mm.flush()
        
        processed_count += current_batch_size
        print(f"  Processed {processed_count}/{total_rows}...", end='\r')

    # Final flush
    parents_mm.flush()
    daughters_mm.flush()
    
    # =========================================================================
    # PHASE 4: SAVE METADATA
    # =========================================================================
    scaler_info = {
        "kinematic_features": ["energy", "x", "y", "z", "px", "py", "pz", "t"],
        "data_min": global_min.tolist(),
        "data_max": global_max.tolist(),
        "pdg_categories": [int(p) for p in sorted_pdgs]
    }
    
    with open(f"{folder}{output_prefix}_scalers.json", "w") as f:
        json.dump(scaler_info, f, indent=4)
        
    parent_scaler_info = {
        "kinematic_features": ["x", "y", "z"],
        "data_min": parent_global_min.tolist(),
        "data_max": parent_global_max.tolist(),
    }
    
    with open(f"{folder}{output_prefix}_parent_scalers.json", "w") as f:
        json.dump(parent_scaler_info, f, indent=4)
        
    print(f"\n\nSuccess! Data saved to {folder}")

if __name__ == "__main__":
    process_bib_data_robust(
        "/global/cfs/cdirs/atlas/rcocasal/MuonCollider/work/genML/data/raw/testML3.csv", 
        output_prefix="bib_data", 
        folder="data/processed/large/",
        chunk_size=5000 
    )
    