import pandas as pd
import numpy as np

def standardize_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes play direction to always be left-to-right.
    If play_direction is 'left', we reflect x and y coordinates.
    """
    # NFL field is 0-120 yards long, 0-53.3 yards wide.
    # If moving left, x decreases. We want x to increase.
    # So new_x = 120 - x
    # new_y = 53.3 - y
    # new_dir = (dir + 180) % 360
    # new_o = (o + 180) % 360
    
    mask = df['play_direction'] == 'left'
    
    df.loc[mask, 'x'] = 120 - df.loc[mask, 'x']
    df.loc[mask, 'y'] = 53.3 - df.loc[mask, 'y']
    
    # Adjust orientation and direction
    if 'dir' in df.columns:
        df.loc[mask, 'dir'] = (df.loc[mask, 'dir'] + 180) % 360
    if 'o' in df.columns:
        df.loc[mask, 'o'] = (df.loc[mask, 'o'] + 180) % 360
        
    # Also adjust ball landing if present
    if 'ball_land_x' in df.columns:
        df.loc[mask, 'ball_land_x'] = 120 - df.loc[mask, 'ball_land_x']
        df.loc[mask, 'ball_land_y'] = 53.3 - df.loc[mask, 'ball_land_y']
        
    return df

def calculate_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes velocity components from speed and direction.
    """
    # NFL data: 0 = North (positive y), 90 = East (positive x).
    # Standard math: 0 = East, 90 = North.
    # Conversion: math_angle = 90 - nfl_angle
    
    if 'dir' in df.columns and 's' in df.columns:
        df['dir_rad'] = np.radians(90 - df['dir'])
        df['vx'] = df['s'] * np.cos(df['dir_rad'])
        df['vy'] = df['s'] * np.sin(df['dir_rad'])
        
    return df

def calculate_relative_to_landing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes features relative to the ball landing spot.
    """
    if 'ball_land_x' not in df.columns:
        return df
        
    df['dist_to_land'] = np.sqrt((df['x'] - df['ball_land_x'])**2 + (df['y'] - df['ball_land_y'])**2)
    
    dx = df['ball_land_x'] - df['x']
    dy = df['ball_land_y'] - df['y']
    df['angle_to_land'] = np.arctan2(dy, dx)
    
    return df

def process_features(features: np.ndarray, targets: np.ndarray, meta: dict) -> tuple:
    """
    Transform function to be used by the Dataset class.
    Args:
        features: (N, 6) array [x, y, s, a, dir, o]
        targets: (M, 2) array [x, y]
        meta: dict containing metadata
    Returns:
        processed_features, processed_targets
    """
    # This function operates on numpy arrays for a single sample
    # But our logic above was for pandas DataFrames.
    # We should probably move the logic to the Dataset class or convert here.
    # For efficiency, let's keep the pandas logic in the Dataset class or a preprocessing step.
    
    # However, since the Dataset class currently returns raw values, 
    # let's assume the Dataset calls a pandas-based transform BEFORE converting to numpy.
    pass

def encode_roles(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encodes player roles.
    Common roles: 'Passer', 'Targeted Receiver', 'Defensive Coverage', 'Other Route Runner', 'Pass Rusher'
    """
    if 'player_role' not in df.columns:
        return df
        
    # Get dummies
    roles = pd.get_dummies(df['player_role'], prefix='role')
    
    # Concatenate
    df = pd.concat([df, roles], axis=1)
    
    # Ensure all expected columns exist (for consistency across batches)
    # We might need a fixed set of roles to ensure consistent columns
    expected_roles = ['role_Passer', 'role_Targeted Receiver', 'role_Defensive Coverage', 
                      'role_Other Route Runner', 'role_Pass Rusher']
                      
    for role in expected_roles:
        if role not in df.columns:
            df[role] = 0
            
    return df

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering steps to a DataFrame.
    """
    df = standardize_direction(df)
    df = calculate_kinematics(df)
    df = calculate_relative_to_landing(df)
    df = encode_roles(df)
    
    # Normalize/Scale features could happen here too
    # For now, we return the engineered dataframe
    return df

