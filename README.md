# Comparative-Analysis-of-LSTM-Variants-for-Time-Series-Forecasting
Comparison of 5 LSTMs for air pollution prediction 

This project was developed as part of the **"Advanced Machine Learning Techniques"** course. The primary objective is to evaluate and compare different Deep Learning architectures for predicting multivariate time-series data, using air pollution (PM2.5) as the case study.

## Project Overview
The study focuses on the **Many-to-One** forecasting approach, where a 24-hour sliding window of 8 meteorological and pollution variables is used to predict the pollution level of the following hour.

### Dataset
The models were trained on a dataset containing **43,800 hourly samples** (5 years of data), including:
* Pollution (PM2.5)
* Dew Point
* Temperature
* Pressure
* Wind Direction & Speed
* Snow & Rain

Source: [Kaggle - LSTM Multivariate Dataset](https://www.kaggle.com/datasets/rupakroy/lstm-datasets-multivariate-univariate)

## Architectures Compared
Five different neural network architectures were built and trained from scratch:
1. **Vanilla LSTM:** The standard Long Short-Term Memory network.
2. **Bidirectional LSTM (BiLSTM):** Processes sequences in both forward and backward directions.
3. **GRU (Gated Recurrent Unit):** A streamlined version of LSTM with fewer parameters.
4. **Conv1D + LSTM:** Combines Convolutional layers for feature extraction with LSTM for temporal dependencies.
5. **Attention-based LSTM:** Implements an attention mechanism to weigh the importance of different time steps.

## Experimental Results (50 Epochs)
The models were evaluated based on **MSE**, **MAE**, and **R² Score**. All models achieved an R² Score > 0.93, with the following highlights:

| Model | R² Score | Training Time (sec) |
| :--- | :---: | :---: |
| **LSTM Classic** | **0.9381** | 102.73 |
| **ConvLSTM** | 0.9375 | 186.10 |
| **BiLSTM** | 0.9364 | 198.05 |
| **Attention LSTM** | 0.9349 | 133.23 |
| **GRU** | 0.9336 | 114.63 |

## Key Findings
* **Efficiency vs Complexity:** The Vanilla LSTM outperformed more complex variants in both accuracy and speed for this specific dataset.
* **BiLSTM Overhead:** Bidirectional processing nearly doubled the training time without a significant gain in R² score.
* **Attention Mechanism:** While complex, the Attention model required more epochs to stabilize its weights compared to simpler architectures.

## How to Run
1. Clone the repository.
2. Ensure you have `tensorflow`, `pandas`, `numpy`, and `scikit-learn` installed.
3. Run the Python script to see the comparative table.
