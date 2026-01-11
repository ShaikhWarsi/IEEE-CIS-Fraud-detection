# IEEE-CIS Fraud Detection: XGBoost "Magic" Model

This project implements a high-performance fraud detection pipeline using XGBoost, focusing on advanced feature engineering techniques (the "Magic" features) to identify unique credit card behavior in the IEEE-CIS Fraud Detection dataset.

## Project Overview

The core of this solution is the identification of unique cardholders (UIDs) and the aggregation of their transaction behavior over time. By normalizing temporal features and analyzing transaction patterns, the model can effectively distinguish between legitimate users and fraudulent actors.

### Key Features
- **D-Column Normalization**: Converting relative time deltas to absolute points in time for stability.
- **Cardholder UID Creation**: Combining multiple card and address features to track individual credit cards.
- **Advanced Encodings**:
  - Frequency Encoding for high-cardinality features.
  - Group Aggregations (Mean, Std, Nunique) based on cardholder UIDs.
- **Optimized Pipeline**: Uses `pd.concat` to avoid DataFrame fragmentation and improve performance.

## Kaggle Results

Below is the result of our model performance on the Kaggle leaderboard:

![Kaggle Result](result.jpeg)

## Getting Started

### Prerequisites
- Python 3.x
- pandas
- numpy
- xgboost
- scikit-learn

### Usage
1. Place the competition datasets (`train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`) in the root directory.
2. Run the training script:
   ```bash
   python xgb_magic_model.py
   ```
3. The script will generate a `submission_xgb_magic.csv` file ready for Kaggle submission.

## Model Report
A detailed technical report of the model architecture and feature engineering process can be found in [XGBoost_Model_Report.md](XGBoost_Model_Report.md).
