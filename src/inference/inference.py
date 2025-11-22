import sys
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.model import TrajectoryPredictor
from src.features.features import preprocess_dataframe, standardize_direction

def run_inference(data_dir, model_path, output_file="submission.csv", device='cpu'):
    print(f"Running inference on {device}...")
    
    # Load Model
    input_size = 15
    model = TrajectoryPredictor(input_size=input_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load Test Data
    test_input_path = os.path.join(data_dir, "test_input.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    print(f"Loading test data from {test_input_path}...")
    df_in = pd.read_csv(test_input_path)
    df_test = pd.read_csv(test_path)
    
    # Preprocess Input
    print("Preprocessing features...")
    # We need to preserve play_direction for standardization
    # But wait, we standardize INPUT to be left-to-right.
    # The OUTPUT predictions will be in standardized coordinates.
    # We need to convert them BACK to original coordinates if the play was 'left'.
    
    # Store original directions
    direction_map = df_in[['game_id', 'play_id', 'play_direction']].drop_duplicates()
    
    df_in = preprocess_dataframe(df_in)
    
    # Group by play/player
    grouped_in = df_in.groupby(['game_id', 'play_id', 'nfl_id'])
    
    # We need to predict for each row in test.csv
    # But test.csv has multiple frames per player.
    # We should predict once per player-play and then map.
    
    # Get unique keys from test.csv
    test_keys = df_test[['game_id', 'play_id', 'nfl_id']].drop_duplicates()
    
    results = []
    
    print(f"Predicting for {len(test_keys)} player-plays...")
    
    # Iterate through unique keys
    # Optimization: Batch this?
    # For now, simple loop is safer to implement correctly.
    
    for _, row in tqdm(test_keys.iterrows(), total=len(test_keys)):
        g_id, p_id, n_id = row['game_id'], row['play_id'], row['nfl_id']
        
        if (g_id, p_id, n_id) not in grouped_in.groups:
            # Should not happen if data is consistent
            continue
            
        # Get input history
        history = grouped_in.get_group((g_id, p_id, n_id))
        
        # Extract features
        feature_cols = [
            'x', 'y', 's', 'a', 'dir', 'o', 'vx', 'vy', 
            'dist_to_land', 'angle_to_land',
            'role_Passer', 'role_Targeted Receiver', 'role_Defensive Coverage', 
            'role_Other Route Runner', 'role_Pass Rusher'
        ]
        
        # Handle missing columns
        for col in feature_cols:
            if col not in history.columns:
                history[col] = 0.0
                
        features = history[feature_cols].values.astype(np.float32)
        features_tensor = torch.tensor(features).unsqueeze(0).to(device) # (1, Seq_Len, Features)
        
        # Determine how many frames to predict
        # We can find the max frame_id requested for this key in df_test
        req_frames = df_test[
            (df_test['game_id'] == g_id) & 
            (df_test['play_id'] == p_id) & 
            (df_test['nfl_id'] == n_id)
        ]['frame_id'].max()
        
        # Predict
        with torch.no_grad():
            preds = model(features_tensor, future_frames=req_frames) # (1, Future_Len, 2)
            
        preds_np = preds.squeeze(0).cpu().numpy()
        
        # Post-process: Convert back to original coordinates if needed
        # Check direction
        direction = direction_map[
            (direction_map['game_id'] == g_id) & 
            (direction_map['play_id'] == p_id)
        ]['play_direction'].iloc[0]
        
        if direction == 'left':
            # Reflect back
            # x = 120 - x
            # y = 53.3 - y
            preds_np[:, 0] = 120 - preds_np[:, 0]
            preds_np[:, 1] = 53.3 - preds_np[:, 1]
            
        # Store results
        # We need to map frame_id 1..N to preds[0..N-1]
        for i in range(req_frames):
            frame_id = i + 1
            results.append({
                'game_id': g_id,
                'play_id': p_id,
                'nfl_id': n_id,
                'frame_id': frame_id,
                'x': preds_np[i, 0],
                'y': preds_np[i, 1]
            })
            
    # Create DataFrame
    pred_df = pd.DataFrame(results)
    
    # Merge with test.csv to ensure correct order and format
    # test.csv has 'id' column? No, it has game/play/nfl/frame.
    # Actually the sample `test.csv` has `id` column.
    # Let's check the sample test.csv again.
    
    # Merge
    submission = df_test.merge(pred_df, on=['game_id', 'play_id', 'nfl_id', 'frame_id'], how='left')
    
    # Fill missing?
    # If test.csv didn't have x,y, then the merged columns are just x, y
    if 'x' in submission.columns:
        submission['x'] = submission['x'].fillna(0)
        submission['y'] = submission['y'].fillna(0)
    else:
        # If collision happened (unlikely based on file inspection)
        submission['x'] = submission['x_y'].fillna(0)
        submission['y'] = submission['y_y'].fillna(0)
    
    # Select columns
    # The submission format usually requires just the ID and predictions?
    # Or just the rows.
    # Let's look at the sample submission or test.csv structure.
    # test.csv: id, game_id, play_id, nfl_id, frame_id
    # We likely need to submit: id, x, y? Or just replace x, y in test.csv?
    # The prompt says: "Contains the prediction targets as rows... representing each position that needs to be predicted."
    
    # Let's save the full csv with predictions
    final_df = submission[['id', 'game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y']]
    
    print(f"Saving submission to {output_file}...")
    final_df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    data_dir = r"c:\Users\Tanish Singla\Desktop\test1\nfl-big-data-bowl-2026-prediction"
    model_path = "src/models/best_model.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_inference(data_dir, model_path, device=device)
