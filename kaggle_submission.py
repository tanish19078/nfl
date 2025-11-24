import sys
import os

# Fix for protobuf compatibility issues in some Kaggle environments
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

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


# ==========================================
# 3. Main Submission Loop (CSV Based)
# ==========================================

DATA_DIR = "/kaggle/input/nfl-big-data-bowl-2026-prediction"



# Load Data
try:
    # Try to load the input data (history) and the submission template (test)
    # Note: Filenames might vary, checking for likely candidates
    input_path = os.path.join(DATA_DIR, "test_input.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    
    if not os.path.exists(input_path):
        # If test_input.csv doesn't exist, maybe test.csv has everything?
        # But we just saw it missing 'x'.
        print(f"Warning: {input_path} not found.")
        # Fallback: maybe the user renamed it?
        # We'll try to proceed with what we have, but it might fail.
        df_in = pd.read_csv(test_path)
    else:
        print(f"Loading input history from {input_path}...")
        df_in = pd.read_csv(input_path)
        
    print(f"Loading test queries from {test_path}...")
    df_test = pd.read_csv(test_path)
    
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

predictions = []

print("Preprocessing history data...")
# Check if 'x' exists in df_in
if 'x' not in df_in.columns:
    print("Error: 'x' column missing from input data. Cannot proceed.")
    print("Columns:", df_in.columns)
    sys.exit(1)

df_proc = preprocess_dataframe(df_in)

# Correct direction: Extract AFTER preprocessing to ensure we have standardized/filled columns if needed
# Although 'play_direction' is usually in the raw data, extracting it from the processed df 
# ensures we are aligned with any changes (though preprocess usually doesn't change the label itself).
# The user noted: "Because direction_map was created before direction preprocessing."
direction_map = df_proc[['game_id', 'play_id', 'nfl_id', 'play_direction']].drop_duplicates()

# Group history by player
grouped_in = df_proc.groupby(['game_id', 'play_id', 'nfl_id'])

# Get unique players to predict for from df_test
# df_test usually contains the frames we need to predict.
print(f"Predicting for {len(df_test)} rows in test.csv...")

# We iterate over unique (game, play, nfl_id) in df_test
unique_players = df_test[['game_id', 'play_id', 'nfl_id']].drop_duplicates()

for _, row in unique_players.iterrows():
    g, p, n = row['game_id'], row['play_id'], row['nfl_id']
    
    if (g, p, n) not in grouped_in.groups:
        # No history for this player?
        continue
        
    # Get history
    history = grouped_in.get_group((g, p, n))
    
    feature_cols = [
        'x', 'y', 's', 'a', 'dir', 'o', 'vx', 'vy',
        'dist_to_land', 'angle_to_land',
        'role_Passer', 'role_Targeted Receiver', 'role_Defensive Coverage',
        'role_Other Route Runner', 'role_Pass Rusher'
    ]

    for col in feature_cols:
        if col not in history:
            history[col] = 0.0

    features = history[feature_cols].values.astype(np.float32)
    tensor = torch.tensor(features).unsqueeze(0).to(device)

    # Get last frame from history
    if 'frame_id' in history.columns:
        last_frame = history['frame_id'].iloc[-1]
    else:
        last_frame = 0

    # Determine max prediction horizon needed
    req_frames_df = df_test[
        (df_test['game_id'] == g) & 
        (df_test['play_id'] == p) & 
        (df_test['nfl_id'] == n)
    ]
    
    if req_frames_df.empty:
        continue
        
    req_frame_ids = req_frames_df['frame_id'].values
    max_req_frame = req_frame_ids.max()
    max_delta = max_req_frame - last_frame
    
    preds_np = None
    if max_delta > 0:
        with torch.no_grad():
            preds = model(tensor, future_frames=int(max_delta))
        preds_np = preds.squeeze(0).cpu().numpy()

        # Post-process (Reflect back if needed)
        if not direction_map[(direction_map['game_id'] == g) & (direction_map['play_id'] == p)].empty:
            p_dir = direction_map[
                (direction_map['game_id'] == g) & 
                (direction_map['play_id'] == p)
            ]['play_direction'].iloc[0]
            
            if p_dir == 'left':
                preds_np[:, 0] = 120 - preds_np[:, 0]
                # preds_np[:, 1] = 53.3 - preds_np[:, 1] # Do not reflect Y for predictions

    # Build rows for submission
    for f_id in req_frame_ids:
        delta = f_id - last_frame
        val_x, val_y = 0.0, 0.0
        
        if delta > 0:
            idx = int(delta) - 1
            if preds_np is not None and idx < len(preds_np):
                val_x = preds_np[idx, 0]
                val_y = preds_np[idx, 1]
        else:
            # Use history
            hist_row = history[history['frame_id'] == f_id]
            if not hist_row.empty:
                val_x = hist_row['x'].iloc[0]
                val_y = hist_row['y'].iloc[0]
                
                # Reverse standardization for history
                # History was flipped BOTH X and Y in preprocess
                if not direction_map[(direction_map['game_id'] == g) & (direction_map['play_id'] == p)].empty:
                    p_dir = direction_map[
                        (direction_map['game_id'] == g) & 
                        (direction_map['play_id'] == p)
                    ]['play_direction'].iloc[0]
                    
                    if p_dir == 'left':
                        val_x = 120 - val_x
                        val_y = 53.3 - val_y

        predictions.append({
            "game_id": g,
            "play_id": p,
            "nfl_id": n,
            "frame_id": f_id,
            "x": val_x,
            "y": val_y,
        })

pred_df = pd.DataFrame(predictions)
# Merge with df_test to ensure we have all rows and correct order if needed
# But usually submission just needs the rows.
# Let's just save what we predicted.
pred_df.to_csv("submission.csv", index=False)
print("Submission complete. Saved to submission.csv")
