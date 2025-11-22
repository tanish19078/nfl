import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataset import NFLDataset
import torch

def verify_dataset():
    data_dir = r"c:\Users\Tanish Singla\Desktop\test1\nfl-big-data-bowl-2026-prediction"
    print(f"Initializing dataset from {data_dir}...")
    
    # Load only week 1 for speed
    dataset = NFLDataset(data_dir, weeks=[1])
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        features, targets = dataset[0]
        print(f"Sample 0 features shape: {features.shape}")
        print(f"Sample 0 targets shape: {targets.shape}")
        print("Features sample (first row):", features[0])
        print("Targets sample (first row):", targets[0])
        
        # Check for NaNs
        if torch.isnan(features).any():
            print("WARNING: NaNs found in features!")
        if torch.isnan(targets).any():
            print("WARNING: NaNs found in targets!")
            
    else:
        print("Dataset is empty!")

if __name__ == "__main__":
    verify_dataset()
