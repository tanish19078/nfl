import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataset import NFLDataset, collate_fn
from src.data.split import get_train_val_split
from src.models.model import TrajectoryPredictor

def masked_mse_loss(preds, targets, lengths):
    """
    MSE loss masked by sequence length.
    preds: (Batch, Max_Len, 2)
    targets: (Batch, Max_Len, 2)
    lengths: (Batch,)
    """
    # Create mask
    batch_size, max_len, _ = targets.size()
    mask = torch.arange(max_len, device=targets.device).expand(batch_size, max_len) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).expand_as(targets) # (Batch, Max_Len, 2)
    
    loss = (preds - targets) ** 2
    loss = loss * mask.float()
    
    # Sum and divide by number of valid elements
    return loss.sum() / mask.sum()

def train_model(data_dir, weeks=None, epochs=10, batch_size=32, lr=0.001, device='cpu'):
    print(f"Training on {device}...")
    
    # Load Dataset
    if weeks is None:
        print("Loading all weeks...")
    else:
        print(f"Loading weeks: {weeks}")
        
    dataset = NFLDataset(data_dir, weeks=weeks)
    
    # Split
    train_set, val_set = get_train_val_split(dataset, val_split=0.2)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize Model
    # Input size: 15 (x, y, s, a, dir, o, vx, vy, dist, angle, 5 roles)
    input_size = 15 
    model = TrajectoryPredictor(input_size=input_size).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for features, targets, feat_lens, target_lens in progress_bar:
            features = features.to(device)
            targets = targets.to(device)
            target_lens = target_lens.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            # Predict same number of frames as targets (max length in batch)
            future_frames = targets.size(1)
            preds = model(features, future_frames=future_frames)
            
            loss = masked_mse_loss(preds, targets, target_lens)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets, feat_lens, target_lens in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                target_lens = target_lens.to(device)
                
                future_frames = targets.size(1)
                preds = model(features, future_frames=future_frames)
                
                loss = masked_mse_loss(preds, targets, target_lens)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "src/models/best_model.pth")
            print("Saved best model.")
            
    print("Training complete.")

if __name__ == "__main__":
    data_dir = r"c:\Users\Tanish Singla\Desktop\test1\nfl-big-data-bowl-2026-prediction"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Train on Weeks 1-18
    train_model(data_dir, weeks=list(range(1, 19)), epochs=10, device=device)
