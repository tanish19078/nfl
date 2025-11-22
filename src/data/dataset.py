import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import sys
import os

# Add src to path to import features
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features.features import preprocess_dataframe, standardize_direction

class NFLDataset(Dataset):
    def __init__(self, data_dir, weeks=None, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing train/ folder.
            weeks (list): List of week numbers to load (e.g., [1, 2]). If None, loads all.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.data = []
        self.keys = []
        
        if weeks is None:
            weeks = range(1, 19) # Weeks 1-18
            
        self._load_data(weeks)

    def _load_data(self, weeks):
        print(f"Loading data from {self.data_dir}...")
        
        for week in weeks:
            week_str = f"{week:02d}"
            input_file = self.data_dir / "train" / f"input_2023_w{week_str}.csv"
            output_file = self.data_dir / "train" / f"output_2023_w{week_str}.csv"
            
            if not input_file.exists() or not output_file.exists():
                print(f"Warning: Files for week {week} not found. Skipping.")
                continue
                
            print(f"Loading Week {week}...")
            df_in = pd.read_csv(input_file)
            df_out = pd.read_csv(output_file)
            
            # Preprocess Input
            # We need to preserve play_direction for output standardization
            # So we'll create a mapping of (game_id, play_id) -> play_direction
            direction_map = df_in[['game_id', 'play_id', 'play_direction']].drop_duplicates()
            
            # Merge direction to output
            df_out = df_out.merge(direction_map, on=['game_id', 'play_id'], how='left')
            
            # Apply standardization to both
            # Note: standardize_direction expects 'x', 'y', 'play_direction'
            print("Preprocessing input features...")
            df_in = preprocess_dataframe(df_in)
            
            print("Standardizing output targets...")
            df_out = standardize_direction(df_out)
            
            # Grouping
            join_cols = ['game_id', 'play_id', 'nfl_id']
            grouped_in = df_in.groupby(join_cols)
            grouped_out = df_out.groupby(join_cols)
            
            # Find common keys
            in_keys = set(grouped_in.groups.keys())
            out_keys = set(grouped_out.groups.keys())
            common_keys = in_keys.intersection(out_keys)
            
            print(f"Found {len(common_keys)} valid player-plays in Week {week}.")
            
            for key in common_keys:
                self.keys.append(key)
                self.data.append({
                    'input': grouped_in.get_group(key),
                    'output': grouped_out.get_group(key)
                })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        df_in = sample['input']
        df_out = sample['output']
        
        # Feature selection
        # Features: x, y, s, a, dir, o, dist_to_land, angle_to_land, vx, vy, roles
        # We need to make sure these columns exist (created by preprocess_dataframe)
        feature_cols = [
            'x', 'y', 's', 'a', 'dir', 'o', 'vx', 'vy', 
            'dist_to_land', 'angle_to_land',
            'role_Passer', 'role_Targeted Receiver', 'role_Defensive Coverage', 
            'role_Other Route Runner', 'role_Pass Rusher'
        ]
        
        # Handle missing columns if any (e.g. if s/a/dir are missing in raw data, but they should be there)
        for col in feature_cols:
            if col not in df_in.columns:
                df_in[col] = 0.0
                
        features = df_in[feature_cols].values.astype(np.float32)
        
        # Targets: x, y
        target_cols = ['x', 'y']
        targets = df_out[target_cols].values.astype(np.float32)
        
        # Metadata
        meta = {
            'game_id': df_in['game_id'].iloc[0],
            'play_id': df_in['play_id'].iloc[0],
            'nfl_id': df_in['nfl_id'].iloc[0],
        }
        
        if self.transform:
            features, targets = self.transform(features, targets, meta)
            
        return torch.tensor(features), torch.tensor(targets)

def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.
    Pads inputs and targets to the max length in the batch.
    """
    features_list, targets_list = zip(*batch)
    
    # Pad features
    # features_list is a tuple of tensors (Seq_Len, Input_Size)
    # We want to pad dim 0
    features_padded = torch.nn.utils.rnn.pad_sequence(features_list, batch_first=True, padding_value=0.0)
    
    # Pad targets
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets_list, batch_first=True, padding_value=0.0)
    
    # Create masks (optional, but good for loss)
    # We can deduce lengths from the original list
    feature_lengths = torch.tensor([len(f) for f in features_list])
    target_lengths = torch.tensor([len(t) for t in targets_list])
    
    return features_padded, targets_padded, feature_lengths, target_lengths
