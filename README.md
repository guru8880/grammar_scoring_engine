# Grammar Scoring Engine from Speech 🎙️

## Overview
This project implements an end-to-end **Grammar Scoring Engine** that predicts a continuous grammar score (0–5) from spoken audio samples. The solution was developed as part of the **SHL Intern Hiring Assessment** hosted on Kaggle.

The system processes spoken responses, extracts meaningful acoustic features, and applies a machine learning regression model to estimate grammar proficiency.

---

## Dataset
- **Source**: Kaggle – SHL Intern Hiring Assessment  
- **Training samples**: 409 audio files  
- **Test samples**: 197 audio files  
- **Audio format**: `.wav` (45–60 seconds per file)

### Files
- `train.csv` – filenames with grammar scores (labels)  
- `test.csv` – filenames for evaluation  
- `audios/train/` – training audio files  
- `audios/test/` – test audio files  

---

## Methodology

### 1. Audio Preprocessing
- Resampling to 16 kHz
- Silence trimming
- Amplitude normalization

### 2. Feature Engineering
Extracted compact and informative acoustic features:
- MFCCs (mean and standard deviation)
- Spectral centroid
- Spectral bandwidth
- Zero Crossing Rate
- RMS energy  

Each audio sample is represented by **34 numerical features**.

### 3. Model
- **Random Forest Regressor**
- Chosen for robustness on small datasets and ability to model non-linear relationships

### 4. Evaluation Metrics
- **RMSE (Root Mean Squared Error)** – mandatory metric
- **Pearson Correlation** – leaderboard metric

---

## Results
- Training and validation RMSE computed
- Pearson correlation used to assess prediction alignment
- Visualization of true vs predicted grammar scores included for interpretability

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/grammar-scoring-engine.git
   cd grammar-scoring-engine
