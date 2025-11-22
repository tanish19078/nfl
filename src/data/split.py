import torch
from torch.utils.data import Subset
from .dataset import NFLDataset
import numpy as np

def get_train_val_split(dataset: NFLDataset, val_split=0.2, seed=42):
    """
    Splits the dataset into training and validation sets based on game_id.
    Ensures that all plays from the same game end up in the same split to prevent leakage.
    """
    # Get all unique game_ids
    # We need to iterate through the dataset to find game_ids
    # This might be slow if dataset is huge and not loaded in memory.
    # But our dataset class loads everything into memory (self.data list).
    
    print("Splitting dataset by game_id...")
    
    game_ids = set()
    game_id_to_indices = {}
    
    for idx, sample in enumerate(dataset.data):
        # We stored metadata in the sample dict? No, we stored raw dataframes.
        # Let's peek at the input dataframe
        g_id = sample['input']['game_id'].iloc[0]
        game_ids.add(g_id)
        
        if g_id not in game_id_to_indices:
            game_id_to_indices[g_id] = []
        game_id_to_indices[g_id].append(idx)
        
    game_ids = list(game_ids)
    np.random.seed(seed)
    np.random.shuffle(game_ids)
    
    n_val_games = int(len(game_ids) * val_split)
    val_games = set(game_ids[:n_val_games])
    train_games = set(game_ids[n_val_games:])
    
    train_indices = []
    val_indices = []
    
    for g_id in train_games:
        train_indices.extend(game_id_to_indices[g_id])
        
    for g_id in val_games:
        val_indices.extend(game_id_to_indices[g_id])
        
    print(f"Total games: {len(game_ids)}")
    print(f"Train games: {len(train_games)}, Val games: {len(val_games)}")
    print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    
    return train_set, val_set
