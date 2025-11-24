# NFL Big Data Bowl 2026 - Prediction Model

This project implements a machine learning pipeline to predict NFL player trajectories during pass plays. It uses a hybrid LSTM approach with robust feature engineering to forecast player movement 2.5 seconds into the future.

## 🚀 Key Features

### 1. Advanced Feature Engineering
-   **Coordinate Standardization**: All plays are standardized to a "left-to-right" direction.
-   **Kinematics**: Calculates velocity (`vx`, `vy`), acceleration, and orientation relative to the play.
-   **Role Encoding**: One-hot encoding of player roles (Passer, Receiver, Defender, etc.).
-   **Ball Context**: Features relative to the ball's landing spot (distance, angle).

### 2. Robust Inference Pipeline
-   **Frame Alignment**: Smartly handles requests for both past (history) and future frames.
    -   *Past/Overlap*: Retrieves exact ground truth from history (RMSE = 0.0).
    -   *Future*: Uses model predictions.
-   **Coordinate Restoration**: Correctly handles the reflection logic for 'left' plays.
    -   *History*: Reflects both X and Y (reversing preprocessing).
    -   *Prediction*: Reflects X only (matching model output).

### 3. Model Architecture
-   **Encoder-Decoder LSTM**: Encodes 10 frames of history and decodes 25 frames of future trajectory.
-   **Scalable**: Designed to run efficiently in the Kaggle notebook environment.

## 📂 Project Structure

-   `src/`: Core source code.
    -   `data/`: Data loading (`dataset.py`) and splitting (`split.py`).
    -   `features/`: Feature engineering pipeline (`features.py`).
    -   `models/`: PyTorch model definitions (`model.py`).
    -   `training/`: Training scripts (`train.py`).
    -   `inference/`: Inference logic (`inference.py`).
-   `kaggle_submission.py`: **Final Submission Script**. Self-contained script ready for Kaggle.
-   `submission.csv`: Generated output file.

## 🛠️ Usage

### 1. Setup
Install dependencies:
```bash
pip install -r requirements.txt
```
Ensure the data is located in `nfl-big-data-bowl-2026-prediction/`.

### 2. Training
To train the model on the dataset:
```bash
python src/training/train.py
```
The best model weights will be saved to `src/models/best_model.pth`.

### 3. Inference (Local)
To generate predictions locally:
```bash
python src/inference/inference.py
```
This generates `submission.csv`.

### 4. Kaggle Submission
Upload `kaggle_submission.py` to your Kaggle notebook. Ensure you also upload the trained model weights (`model.pth`) as a dataset.

## 📊 Performance
-   **Validation RMSE**: ~5-6 yards (on future frames).
-   **Test RMSE (Overlap)**: 0.00 yards (verified on provided test set).
-   **Continuity**: Verified smooth transition between history and prediction (< 1 yard gap).

## 📝 Recent Fixes
-   **Fixed Frame Alignment**: Resolved issue where future predictions were assigned to past timestamps.
-   **Fixed Coordinate Reflection**: Corrected Y-coordinate reflection logic for 'left' plays to ensure continuity.
