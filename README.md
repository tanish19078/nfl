# NFL Big Data Bowl 2026 - Prediction Model

This project implements a hybrid LSTM-GNN model to predict NFL player trajectories during pass plays.

## Project Structure

- `src/`: Source code.
    - `data/`: Data loading and splitting (`dataset.py`, `split.py`).
    - `features/`: Feature engineering pipeline (`features.py`).
    - `models/`: Model architecture (`model.py`, `architecture.py`).
    - `training/`: Training scripts (`train.py`).
    - `inference/`: Inference scripts (`inference.py`).

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure data is placed in `nfl-big-data-bowl-2026-prediction/`.

## Usage

### 1. Train Model
Train the hybrid LSTM-GNN model on the real dataset:
```bash
python src/training/train.py
```
This will load data from `nfl-big-data-bowl-2026-prediction/train/`, process features, and train the model. The best model is saved to `src/models/best_model.pth`.

### 2. Inference
Run inference on the test set and generate `submission.csv`:
```bash
python src/inference/inference.py
```

## Model Architecture

- **Encoder**: LSTM encodes the past 10 frames of player movement.
- **Interaction Module**: GNN (Graph Attention) captures spatial interactions between players (Offense vs Defense).
- **Decoder**: LSTM decodes the future 25 frames (2.5 seconds) of trajectory.
