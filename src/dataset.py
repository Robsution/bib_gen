import torch
import numpy as np
from torch.utils.data import Dataset
import os

class BibDataset(Dataset):
    def __init__(self, data_dir, prefix="bib_data", min_n=0, max_n=np.inf):
        super().__init__()
        """
        Args:
            data_dir (str): Path to folder containing .npy files
            prefix (str): Prefix of the files (e.g., 'bib_data')
        """
        parents_path = os.path.join(data_dir, f"{prefix}_parents.npy")
        daughters_path = os.path.join(data_dir, f"{prefix}_daughters.npy")
        
        # self.parents = np.load(parents_path, mmap_mode='r')
        # self.daughters = np.load(daughters_path, mmap_mode='r')
        
        self.parents = np.load(parents_path, mmap_mode='r')
        self.daughters = np.load(daughters_path, mmap_mode='r')
        
        self.n_samples = self.parents.shape[0]
        

        filtering = not (min_n == 0 and max_n == np.inf)

        if not filtering:
            self.parents = torch.tensor(np.array(self.parents), dtype=torch.float32)
            self.daughters = torch.tensor(np.array(self.daughters), dtype=torch.float32)
        else:
            print(f"Filtering dataset for families with [{min_n}, {max_n}] daughters...")
            
            filtered_parents = []
            filtered_daughters = []
            
            BATCH_SIZE = 20000  # Adjust this based on your RAM availability
            num_samples = len(self.parents)
            
            for i in range(0, num_samples, BATCH_SIZE):
                # Slice a manageable chunk of your Python lists
                p_batch = self.parents[i : i + BATCH_SIZE]
                d_batch = self.daughters[i : i + BATCH_SIZE]
                
                # Vectorized math on just this small batch
                d_batch_arr = np.array(d_batch)
                true_counts = np.sum(d_batch_arr[:, :, 0], axis=1)
                
                # Boolean mask for the batch
                mask = (true_counts >= min_n) & (true_counts <= max_n)
                
                # Extract passing elements and convert to list elements temporarily
                filtered_parents.extend(np.array(p_batch)[mask])
                filtered_daughters.extend(d_batch_arr[mask])
                
            # Convert the final consolidated lists into PyTorch tensors
            self.parents = torch.tensor(np.array(filtered_parents), dtype=torch.float32)
            self.daughters = torch.tensor(np.array(filtered_daughters), dtype=torch.float32)
        
        self.min_daughters = min_n
        self.max_daughters = max_n
        print(f"Success! Filtered down to {len(self.parents)} families.")

    def __len__(self):
        return len(self.parents)

    def __getitem__(self, idx):
        
        parent = torch.from_numpy(np.array(self.parents[idx])).float()
        raw_daughters = np.array(self.daughters[idx]) 
        
        # Split Mask and Features
        mask = torch.from_numpy(raw_daughters[:, 0:1]).float()
        features = torch.from_numpy(raw_daughters[:, 1:]).float()
        
        true_n = torch.sum(mask) / self.max_daughters
        
        return parent, features, mask, true_n