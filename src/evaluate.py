import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from math import sqrt

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import NFLDataset, collate_fn
from data.split import get_train_val_split
from models.model import TrajectoryPredictor

def compute_ade(x_true, y_true, x_pred, y_pred):
    """
    ADE = average L2 distance across all time steps.
    """
    dist = torch.sqrt((x_true - x_pred)**2 + (y_true - y_pred)**2)
    return torch.mean(dist)

def compute_fde(x_true, y_true, x_pred, y_pred):
    """
    FDE = L2 distance at the final predicted frame only.
    """
    # x_true: (Batch, Seq)
    # We want the last frame for each sequence in the batch
    # But sequences are padded. We need to know the length.
    # However, for this simple calculation, let's assume we can just take the last element 
    # if we unpad or if we use the lengths.
    
    # Actually, let's compute it per sample to be safe with lengths
    fde_sum = 0.0
    count = 0
    
    batch_size = x_true.size(0)
    for i in range(batch_size):
        # We don't have lengths passed here easily unless we change signature
        # But wait, we can pass lengths or just compute on the whole padded sequence 
        # if we masked it.
        # For simplicity in this script, let's just take the last frame of the *target* 
        # assuming the target is the ground truth length.
        # But the tensor is padded.
        pass
    
    # Let's use the masked approach if we have lengths.
    # For now, let's just implement a simple version that assumes we handle it in the loop.
    return 0.0

def evaluate():
    data_dir = r"c:\Users\Tanish Singla\Desktop\test1\nfl-big-data-bowl-2026-prediction"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Evaluating on {device}...")
    
    # Load Dataset (Weeks 1-18)
    # Note: If training was only on Week 1, we should probably evaluate on Week 1 validation 
    # to get meaningful results for *that* model.
    # But the user wants "final" results.
    # Let's load all weeks.
    dataset = NFLDataset(data_dir, weeks=list(range(1, 19)))
    
    # Split
    _, val_set = get_train_val_split(dataset, val_split=0.2)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Load Model
    input_size = 15
    model = TrajectoryPredictor(input_size=input_size).to(device)
    model_path = os.path.join(os.path.dirname(__file__), 'models/best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    total_ade = 0.0
    total_fde = 0.0
    total_rmse_sq = 0.0
    total_samples = 0
    total_frames = 0
    
    with torch.no_grad():
        for features, targets, feat_lens, target_lens in tqdm(val_loader, desc="Evaluating"):
            features = features.to(device)
            targets = targets.to(device)
            target_lens = target_lens.to(device)
            
            future_frames = targets.size(1)
            preds = model(features, future_frames=future_frames)
            
            # Unpad/Mask
            # Iterate through batch
            for i in range(features.size(0)):
                length = target_lens[i]
                
                pred_seq = preds[i, :length]
                true_seq = targets[i, :length]
                
                x_pred = pred_seq[:, 0]
                y_pred = pred_seq[:, 1]
                x_true = true_seq[:, 0]
                y_true = true_seq[:, 1]
                
                # ADE
                dist = torch.sqrt((x_true - x_pred)**2 + (y_true - y_pred)**2)
                total_ade += torch.sum(dist).item()
                
                # FDE
                fde = torch.sqrt((x_true[-1] - x_pred[-1])**2 + (y_true[-1] - y_pred[-1])**2)
                total_fde += fde.item()
                
                # RMSE (Competition) components
                # (dx^2 + dy^2) / 2
                mse_comp = ((x_true - x_pred)**2 + (y_true - y_pred)**2) / 2.0
                total_rmse_sq += torch.sum(mse_comp).item()
                
                total_samples += 1
                total_frames += length.item()
                
    avg_ade = total_ade / total_frames
    avg_fde = total_fde / total_samples
    rmse = sqrt(total_rmse_sq / total_frames)
    
    print("=== Evaluation Summary ===")
    print(f"ADE:  {avg_ade:.4f} yards")
    print(f"FDE:  {avg_fde:.4f} yards")
    print(f"RMSE (Competition Metric): {rmse:.4f} yards")

if __name__ == "__main__":
    evaluate()
