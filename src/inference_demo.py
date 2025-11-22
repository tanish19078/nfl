import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.model import TrajectoryPredictor
from src.data.dataset import NFLDataset

def run_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    # Input size must match training: 15 features
    model = TrajectoryPredictor(input_size=15).to(device)
    
    model_path = os.path.join("src", "models", "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Run train.py first.")
        return
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    # Load a sample from dataset
    # We'll load just Week 5 for demo purposes to be fast
    data_dir = r"c:\Users\Tanish Singla\Desktop\test1\nfl-big-data-bowl-2026-prediction"
    print("Loading Week 5 data for inference demo...")
    dataset = NFLDataset(data_dir, weeks=[5])
    
    if len(dataset) == 0:
        print("No samples found in dataset.")
        return
        
    print(f"Dataset loaded with {len(dataset)} samples.")
    
    # Get a sample
    # dataset[i] returns (features, targets)
    # features: (Seq_Len, 15)
    # targets: (Pred_Len, 2)
    
    sample_idx = 0
    features, targets = dataset[sample_idx]
    
    # Add batch dimension
    features = features.unsqueeze(0).to(device) # (1, Seq_Len, 15)
    targets = targets.unsqueeze(0).to(device)   # (1, Pred_Len, 2)
    
    print(f"Input shape: {features.shape}")
    print(f"Target shape: {targets.shape}")
    
    with torch.no_grad():
        # Predict
        future_frames = targets.size(1)
        preds = model(features, future_frames=future_frames)
        
    # Move to CPU for plotting
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    features = features.cpu().numpy()
    
    # Plotting
    # We want to show the history (from features) and the prediction vs truth
    # Features: x, y are indices 0, 1
    
    history_x = features[0, :, 0]
    history_y = features[0, :, 1]
    
    pred_x = preds[0, :, 0]
    pred_y = preds[0, :, 1]
    
    true_x = targets[0, :, 0]
    true_y = targets[0, :, 1]
    
    plt.figure(figsize=(12, 8))
    
    # Plot history
    plt.plot(history_x, history_y, 'b.-', label='History', alpha=0.7)
    # Mark end of history
    plt.plot(history_x[-1], history_y[-1], 'bo', markersize=8)
    
    # Plot Ground Truth
    plt.plot(true_x, true_y, 'g.-', label='Ground Truth', alpha=0.7)
    
    # Plot Prediction
    plt.plot(pred_x, pred_y, 'r.--', label='Prediction', alpha=0.7)
    
    plt.title(f"Trajectory Prediction Sample {sample_idx}")
    plt.xlabel("X (yards)")
    plt.ylabel("Y (yards)")
    plt.legend()
    plt.grid(True)
    
    output_file = "inference_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    run_inference()
