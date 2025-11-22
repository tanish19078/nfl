import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataset import NFLDataset
from src.data.split import get_train_val_split

def verify_split():
    data_dir = r"c:\Users\Tanish Singla\Desktop\test1\nfl-big-data-bowl-2026-prediction"
    print(f"Initializing dataset from {data_dir}...")
    
    # Load only week 1 for speed
    dataset = NFLDataset(data_dir, weeks=[1])
    
    if len(dataset) == 0:
        print("Dataset is empty!")
        return

    train_set, val_set = get_train_val_split(dataset, val_split=0.2)
    
    print(f"Train set size: {len(train_set)}")
    print(f"Val set size: {len(val_set)}")
    
    # Verify no leakage
    train_games = set()
    val_games = set()
    
    print("Checking for leakage...")
    # Subset doesn't expose the underlying data directly easily without iterating
    # But we can access .dataset and .indices
    
    for idx in train_set.indices:
        sample = dataset.data[idx]
        train_games.add(sample['input']['game_id'].iloc[0])
        
    for idx in val_set.indices:
        sample = dataset.data[idx]
        val_games.add(sample['input']['game_id'].iloc[0])
        
    print(f"Train games: {sorted(list(train_games))}")
    print(f"Val games: {sorted(list(val_games))}")
    
    intersection = train_games.intersection(val_games)
    if len(intersection) > 0:
        print(f"ERROR: Leakage detected! Games in both: {intersection}")
    else:
        print("SUCCESS: No leakage detected. Train and Val games are disjoint.")

if __name__ == "__main__":
    verify_split()
