import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features.features import preprocess_dataframe, standardize_direction

def verify_pandas_logic():
    data_dir = Path(r"c:\Users\Tanish Singla\Desktop\test1\nfl-big-data-bowl-2026-prediction")
    week = 1
    week_str = f"{week:02d}"
    
    input_file = data_dir / "train" / f"input_2023_w{week_str}.csv"
    output_file = data_dir / "train" / f"output_2023_w{week_str}.csv"
    
    print(f"Reading {input_file}...")
    if not input_file.exists():
        print("File not found.")
        return

    df_in = pd.read_csv(input_file)
    df_out = pd.read_csv(output_file)
    
    print(f"Input shape: {df_in.shape}")
    print(f"Output shape: {df_out.shape}")
    
    # Logic from dataset.py
    direction_map = df_in[['game_id', 'play_id', 'play_direction']].drop_duplicates()
    df_out = df_out.merge(direction_map, on=['game_id', 'play_id'], how='left')
    
    print("Preprocessing...")
    df_in = preprocess_dataframe(df_in)
    df_out = standardize_direction(df_out)
    
    print("Grouping...")
    join_cols = ['game_id', 'play_id', 'nfl_id']
    grouped_in = df_in.groupby(join_cols)
    grouped_out = df_out.groupby(join_cols)
    
    in_keys = set(grouped_in.groups.keys())
    out_keys = set(grouped_out.groups.keys())
    common_keys = in_keys.intersection(out_keys)
    
    print(f"Found {len(common_keys)} valid player-plays.")
    
    if len(common_keys) > 0:
        key = list(common_keys)[0]
        print(f"Sample Key: {key}")
        sample_in = grouped_in.get_group(key)
        sample_out = grouped_out.get_group(key)
        
        print("Sample Input Head:")
        print(sample_in[['x', 'y', 's', 'dir', 'play_direction']].head())
        print("Sample Output Head:")
        print(sample_out[['x', 'y']].head())

if __name__ == "__main__":
    verify_pandas_logic()
