import numpy as np
from src.dataset import BibDataset 
import argparse

def verify_dataset(min_daughters = 0, max_daughters = np.inf):
    # Replace 'data_dir' with your actual path from your config
    data_dir = "/global/cfs/cdirs/atlas/rcocasal/MuonCollider/work/genML/data/processed/medium"
    
    print("Initializing dataset...")
    dataset = BibDataset(data_dir, min_n = min_daughters, max_n = max_daughters) 
    
    print(f"\nTotal families found: {len(dataset)}")
    
    if len(dataset) == 0:
        print(f"ERROR: No families with number of daughters within [{min_daughters}, {max_daughters}] found in the data!")
        return

    print("\n--- Checking the first 3 samples ---")
    for i in range(min(3, len(dataset))):
        parent, features, mask, true_n = dataset[i]
        print(f"Sample {i}:")
        print(f"  Parent shape: {parent.shape}")
        print(f"  Features shape: {features.shape} (Should be [2, feature_dim])")
        print(f"  Mask shape: {mask.shape} (Should be [2, 1])")
        print(f"  Mask values: {mask.flatten().tolist()} (Should be [1.0, 1.0])")
        print(f"  Scaled true_n: {true_n.item()} (Should be 1.0)")
        print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Properties Verification Tool")
    
    parser.add_argument("--min", type=int, default=0, help="Minimum number of daughters in each family")
    parser.add_argument("--max", type=int, default=np.inf, help="Maximum number of daughter in each family")
    
    args = parser.parse_args()
    verify_dataset(min_daughters = args.min, max_daughters = args.max)