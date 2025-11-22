import torch
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataset import NFLDataset, collate_fn
from src.data.split import get_train_val_split
from src.models.model import TrajectoryPredictor

def calculate_metrics(preds, targets, lengths):
    """
    Calculate ADE and FDE.
    preds: (Batch, Max_Len, 2)
    targets: (Batch, Max_Len, 2)
    lengths: (Batch,)
    """
    batch_size = preds.size(0)
    ade_sum = 0.0
    fde_sum = 0.0
    total_samples = 0
    
    for i in range(batch_size):
        length = lengths[i]
        if length == 0:
            continue
            
        pred_seq = preds[i, :length]
        target_seq = targets[i, :length]
        
        # Euclidean distance for each step
        dist = torch.norm(pred_seq - target_seq, dim=1) # (Length,)
        
        ade = dist.mean().item()
        fde = dist[-1].item()
        
        ade_sum += ade
        fde_sum += fde
        total_samples += 1
        
    return ade_sum, fde_sum, total_samples

def evaluate_model(data_dir, model_path, weeks=None, batch_size=32, device='cpu'):
    print(f"Evaluating on {device}...")
    
    # Load Dataset
    # We'll use the validation set from the split
    dataset = NFLDataset(data_dir, weeks=weeks)
    _, val_set = get_train_val_split(dataset, val_split=0.2)
    
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Load Model
    input_size = 15
    model = TrajectoryPredictor(input_size=input_size).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded.")
    else:
        print("Model not found!")
        return
        
    model.eval()
    
    total_ade = 0.0
    total_fde = 0.0
    total_count = 0
    
    with torch.no_grad():
        for features, targets, feat_lens, target_lens in tqdm(val_loader, desc="Evaluating"):
            features = features.to(device)
            targets = targets.to(device)
            target_lens = target_lens.to(device)
            
            future_frames = targets.size(1)
            preds = model(features, future_frames=future_frames)
            
            ade, fde, count = calculate_metrics(preds, targets, target_lens)
            
            total_ade += ade
            total_fde += fde
            total_count += count
            
    avg_ade = total_ade / total_count if total_count > 0 else 0
    avg_fde = total_fde / total_count if total_count > 0 else 0
    
    print(f"\nEvaluation Results:")
    print(f"Average Displacement Error (ADE): {avg_ade:.4f}")
    print(f"Final Displacement Error (FDE): {avg_fde:.4f}")
    
    return avg_ade, avg_fde

if __name__ == "__main__":
    data_dir = r"c:\Users\Tanish Singla\Desktop\test1\nfl-big-data-bowl-2026-prediction"
    model_path = r"c:\Users\Tanish Singla\Desktop\test1\src\models\best_model.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Evaluate on a subset of weeks or all? 
    # Let's evaluate on the same weeks we trained on (validation split handles the unseen data)
    evaluate_model(data_dir, model_path, weeks=list(range(1, 19)), device=device)
