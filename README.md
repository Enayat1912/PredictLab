# 🧠 PredictLab – Machine Learning Dashboard

[![Streamlit App](https://img.shields.io/badge/Live_App-mlpredictlab.streamlit.app-brightgreen?logo=streamlit)](https://mlpredictlab.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

**PredictLab** is an interactive Streamlit web app that lets you upload a CSV dataset, train and compare machine learning models, visualize metrics, and generate predictions — all directly from your browser.  
It’s designed for both beginners and data science enthusiasts who want a quick, no-code way to experiment with machine learning.

---

## ✨ Features

- 📂 Upload any CSV dataset (last column = target)
- ⚙️ Automatic preprocessing (handles encoding, scaling, and missing values)
- 🤖 Multiple ML algorithms supported:
  - Logistic Regression  
  - Random Forest  
  - Support Vector Machine (SVM)  
  - K-Nearest Neighbors (KNN)  
  - Gradient Boosting  
  - And more
- 📊 Built-in visualizations:
  - Confusion Matrix  
  - ROC & Precision-Recall Curves  
  - Feature Importance  
  - Model Coefficients
- 🔮 Prediction modes:
  - Single input prediction  
  - Batch CSV upload for predictions
- 💾 Download options:
  - Predictions as CSV  
  - Trained model as `.pkl` file
- 🌐 100% browser-based — no code required!

---

## 🚀 Live Demo

👉 **Try it live:** [https://mlpredictlab.streamlit.app](https://mlpredictlab.streamlit.app)

Upload a CSV, select a model, train it, and visualize performance instantly.

---

## 🧩 Installation (Run Locally)

```bash
# Clone the repository
git clone https://github.com/Enayat1912/PredictLab.git
cd PredictLab

# Create a virtual environment (optional but recommended)
python -m venv .venv

# Activate the environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
