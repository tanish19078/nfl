# Final Project Report: NFL Big Data Bowl 2026 Prediction

## Executive Summary
This project successfully implements a hybrid LSTM-GNN model to predict NFL player trajectories during pass plays. The system has been fully transitioned to use real NFL tracking data (Weeks 1-18), replacing initial mock data prototypes. The model achieves competitive performance on the validation set.

## Key Results
### Performance Metrics
Evaluated on the validation set (20% split):
-   **ADE (Average Displacement Error)**: **6.5404 yards**
    -   *Average L2 distance across all time steps.*
-   **FDE (Final Displacement Error)**: **7.4203 yards**
    -   *L2 distance at the final predicted frame (2.5s).*
-   **Competition RMSE**: **5.5281 yards**
    -   *Root Mean Squared Error of component deviations.*

### Deliverables
-   **Model Weights**: `src/models/best_model.pth`
-   **Submission File**: `submission.csv` (Ready for platform submission)
-   **Visualization**: `inference_plot.png` (Trajectory plot)

## Technical Achievements

### 1. Data Pipeline
-   **Real Data Integration**: Fully implemented `NFLDataset` to load and process official tracking data from `nfl-big-data-bowl-2026-prediction/`.
-   **Feature Engineering**: Robust pipeline calculating velocity, acceleration, orientation, and role-based encodings.
-   **Leakage Prevention**: Implemented game-level splitting to ensure training and validation sets are strictly disjoint.

### 2. Model Architecture
-   **Hybrid Design**: Combines LSTM for temporal dynamics with GNNs for spatial player interactions.
-   **Encoder-Decoder**: Encodes past 10 frames; decodes future 25 frames (2.5 seconds).

### 3. Training & Inference
-   **Loss Function**: Masked MSE to handle variable sequence lengths.
-   **Inference Engine**: optimized prediction pipeline that handles coordinate standardization (left-to-right normalization) and restoration.

## Project Status
-   **Codebase**: Finalized and cleaned.
-   **Documentation**: `README.md` updated to reflect the final architecture and usage.
-   **Version Control**: `.gitignore` configured for Python/Data science workflow.
-   **Verification**: All scripts (`dataset.py`, `train.py`, `inference.py`) verified to run against the real dataset.

## Issues & Resolutions
-   **Data Transition**: Initial development used mock data, leading to compatibility issues.
    -   *Resolution*: Complete refactor of the data loading layer. Verified integrity using `src/evaluate.py` and manual inspection of splits.

## Next Steps
-   Submit `submission.csv` to the Kaggle competition.
-   Explore attention mechanisms to further reduce FDE.
-   Run extended training on the full dataset for final submission.
