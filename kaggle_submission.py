import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ==========================================
# 1. Model Architecture
# ==========================================
class TrajectoryPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=2, dropout=0.2):
        super(TrajectoryPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully Connected Layer to map hidden state to output coordinates
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, future_frames=0, teacher_forcing_ratio=0.0):
        """
        Args:
            x: Input sequence (Batch, Seq_Len, Features)
            future_frames: Number of future frames to predict
        """
        batch_size = x.size(0)
        
        # Encoder
        _, (hidden, cell) = self.lstm(x)
        
        # Decoder (Simple autoregressive for V1)
        decoder_input = x[:, -1, :] # Last frame of input
        
        preds = []
        curr_hidden = hidden
        curr_cell = cell
        curr_input = decoder_input
        
        for _ in range(future_frames):
            # Run LSTM cell (one step)
            out, (curr_hidden, curr_cell) = self.lstm(curr_input.unsqueeze(1), (curr_hidden, curr_cell))
            
            # Predict position
            pos_pred = self.fc(out.squeeze(1)) # (Batch, 2)
            preds.append(pos_pred)
            
            # Prepare next input (Simple approximation: copy previous features, update x,y)
            next_input = curr_input.clone()
            next_input[:, 0:2] = pos_pred 
            
            curr_input = next_input
            
        return torch.stack(preds, dim=1)

# ==========================================
# 2. Feature Engineering
# ==========================================
def standardize_direction(df: pd.DataFrame) -> pd.DataFrame:
    mask = df['play_direction'] == 'left'
    
    # Standardize x, y
    df.loc[mask, 'x'] = 120 - df.loc[mask, 'x']
    df.loc[mask, 'y'] = 53.3 - df.loc[mask, 'y']
    
    # Adjust orientation and direction
    if 'dir' in df.columns:
        df.loc[mask, 'dir'] = (df.loc[mask, 'dir'] + 180) % 360
    if 'o' in df.columns:
        df.loc[mask, 'o'] = (df.loc[mask, 'o'] + 180) % 360
        
    return df

def calculate_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    if 'dir' in df.columns and 's' in df.columns:
        df['dir_rad'] = np.radians(90 - df['dir'])
        df['vx'] = df['s'] * np.cos(df['dir_rad'])
        df['vy'] = df['s'] * np.sin(df['dir_rad'])
    return df

def calculate_relative_to_landing(df: pd.DataFrame) -> pd.DataFrame:
    if 'ball_land_x' not in df.columns:
        # If ball landing not provided in test, fill 0
        df['dist_to_land'] = 0.0
        df['angle_to_land'] = 0.0
        return df
        
    df['dist_to_land'] = np.sqrt((df['x'] - df['ball_land_x'])**2 + (df['y'] - df['ball_land_y'])**2)
    
    dx = df['ball_land_x'] - df['x']
    dy = df['ball_land_y'] - df['y']
    df['angle_to_land'] = np.arctan2(dy, dx)
    return df

def encode_roles(df: pd.DataFrame) -> pd.DataFrame:
    if 'player_role' not in df.columns:
        # If role not provided, create dummy columns
        expected_roles = ['role_Passer', 'role_Targeted Receiver', 'role_Defensive Coverage', 
                          'role_Other Route Runner', 'role_Pass Rusher']
        for role in expected_roles:
            df[role] = 0
        return df
        
    roles = pd.get_dummies(df['player_role'], prefix='role')
    df = pd.concat([df, roles], axis=1)
    
    expected_roles = ['role_Passer', 'role_Targeted Receiver', 'role_Defensive Coverage', 
                      'role_Other Route Runner', 'role_Pass Rusher']
                      
    for role in expected_roles:
        if role not in df.columns:
            df[role] = 0
            
    return df

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure required columns exist or handle missing
    if 'play_direction' not in df.columns:
        df['play_direction'] = 'right' # Default if missing
        
    df = standardize_direction(df)
    df = calculate_kinematics(df)
    df = calculate_relative_to_landing(df)
    df = encode_roles(df)
    return df

# ==========================================
# 3. Main Submission Loop
# ==========================================

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load Model
# Assuming model.pth is uploaded to /kaggle/input/your-dataset/model.pth
# Update this path to match your Kaggle dataset structure
MODEL_PATH = "/kaggle/input/nfl-model/model.pth" 

input_size = 15
model = TrajectoryPredictor(input_size=input_size).to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model loaded successfully.")
else:
    print(f"Warning: Model not found at {MODEL_PATH}. Using initialized weights (for testing only).")
    
model.eval()

# Initialize Environment
try:
    import kaggle_environments
except ImportError:
    print("kaggle_environments not found. Simulating loop if local.")
    sys.exit(0)

from kaggle_environments import make

# Note: The competition environment name usually uses underscores.
# We will try a few likely candidates.
env_names = ["nfl-big-data-bowl-2026-prediction", "nfl_big_data_bowl_2025", "nfl_big_data_bowl_2026", "nfl-big-data-bowl-2025", "nfl-big-data-bowl-2026", "nfl"]
env = None

for name in env_names:
    try:
        print(f"Attempting to load environment: {name}")
        env = make(name, debug=True)
        print(f"Successfully loaded {name}!")
        break
    except Exception:
        continue
        
if env is None:
    print("Error: Could not load any NFL environment.")
    print("Available environments:")
    # Print available environments to help debug
    try:
        from kaggle_environments import list_environments
        print(list_environments())
    except ImportError:
        print("Could not list environments.")
    sys.exit(1)

iter_test = env.iter_test()

print("Starting prediction loop...")

for (test_df, sample_prediction_df) in iter_test:
    # test_df contains data for the current play(s)
    
    # Store original directions for post-processing
    direction_map = test_df[['game_id', 'play_id', 'nfl_id', 'play_direction']].drop_duplicates()
    
    # Apply features
    df_proc = preprocess_dataframe(test_df)
    
    # Group by player
    grouped = df_proc.groupby(['game_id', 'play_id', 'nfl_id'])
    
    predictions = []
    
    # Get unique players to predict for
    players_to_predict = sample_prediction_df[['game_id', 'play_id', 'nfl_id']].drop_duplicates()
    
    for _, row in players_to_predict.iterrows():
        g_id, p_id, n_id = row['game_id'], row['play_id'], row['nfl_id']
        
        if (g_id, p_id, n_id) not in grouped.groups:
            continue
            
        history = grouped.get_group((g_id, p_id, n_id))
        
        # Extract features
        feature_cols = [
            'x', 'y', 's', 'a', 'dir', 'o', 'vx', 'vy', 
            'dist_to_land', 'angle_to_land',
            'role_Passer', 'role_Targeted Receiver', 'role_Defensive Coverage', 
            'role_Other Route Runner', 'role_Pass Rusher'
        ]
        
        # Handle missing cols
        for col in feature_cols:
            if col not in history.columns:
                history[col] = 0.0
        
        features = history[feature_cols].values.astype(np.float32)
        features_tensor = torch.tensor(features).unsqueeze(0).to(device)
        
        # Determine frames to predict
        req_frames_df = sample_prediction_df[
            (sample_prediction_df['game_id'] == g_id) & 
            (sample_prediction_df['play_id'] == p_id) & 
            (sample_prediction_df['nfl_id'] == n_id)
        ]
        
        if req_frames_df.empty:
            continue
            
        num_frames = len(req_frames_df)
        
        with torch.no_grad():
            preds = model(features_tensor, future_frames=num_frames)
            
        preds_np = preds.squeeze(0).cpu().numpy()
        
        # Post-process (Reflect back if needed)
        p_dir = direction_map[
            (direction_map['game_id'] == g_id) & 
            (direction_map['play_id'] == p_id) &
            (direction_map['nfl_id'] == n_id)
        ]['play_direction'].iloc[0]
        
        if p_dir == 'left':
            preds_np[:, 0] = 120 - preds_np[:, 0]
            preds_np[:, 1] = 53.3 - preds_np[:, 1]
            
        # Fill predictions
        req_frames_df = req_frames_df.sort_values('frame_id')
        
        for i, (idx, req_row) in enumerate(req_frames_df.iterrows()):
            if i < len(preds_np):
                predictions.append({
                    'game_id': g_id,
                    'play_id': p_id,
                    'nfl_id': n_id,
                    'frame_id': req_row['frame_id'],
                    'x': preds_np[i, 0],
                    'y': preds_np[i, 1]
                })
                
    # Create DataFrame
    if predictions:
        pred_df = pd.DataFrame(predictions)
        
        # Efficient update
        pred_df.set_index(['game_id', 'play_id', 'nfl_id', 'frame_id'], inplace=True)
        sample_prediction_df.set_index(['game_id', 'play_id', 'nfl_id', 'frame_id'], inplace=True)
        
        sample_prediction_df.update(pred_df)
        sample_prediction_df.reset_index(inplace=True)
        
        sample_prediction_df.fillna(0, inplace=True)
    else:
        # Fallback if no predictions
        sample_prediction_df['x'] = 0.0
        sample_prediction_df['y'] = 0.0
        
    # Submit
    env.predict(sample_prediction_df)

print("Submission complete.")
