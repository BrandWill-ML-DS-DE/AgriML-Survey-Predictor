# 🌾 AgriML: Agricultural Adoption & Propensity Predictor

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Neural Networks](https://img.shields.io/badge/Model-MLP-red)](https://en.wikipedia.org/wiki/Multi-layer_perceptron)

A predictive modeling pipeline designed to analyze and predict farmer willingness to adopt new agricultural technologies. This system moves beyond standard classification by correcting for **survey non-response bias** and extracting latent features from unstructured qualitative feedback.

---

## 🧠 Advanced Engineering & Statistical Design

This project showcases three critical "Senior-level" data science concepts required for high-stakes decision-making:

### 1. Propensity Score Weighting (IPW)
Real-world surveys are often biased; for example, smaller subsistence farms might be less likely to respond than larger corporate farms. 
* **The Solution:** I trained a Random Forest Classifier to predict the probability of a farmer responding based on demographics.
* **The Impact:** By calculating a $Weight = \frac{1}{Propensity\_Score}$, the final model gives more "voice" to underrepresented groups, ensuring predictions represent the entire population rather than just the vocal minority.



### 2. Hybrid NLP-Tabular Integration
Qualitative "notes" often hold more signal than categorical numbers.
* **The Solution:** The pipeline utilizes an NLP Sentiment Analyzer to extract intent (e.g., "interested" vs. "too expensive") from raw feedback.
* **The Impact:** This transforms unstructured text into a `Sentiment_Feature`, fed directly into the neural network to boost "contextual awareness."

### 3. Multi-Layer Perceptron (MLP) for Non-Linear Logic
While simpler models like Logistic Regression are common in survey analysis, I utilized an **MLP (Neural Network)** to capture the complex, non-linear relationships between soil chemistry ($N, P, K, pH$), social trust levels, and financial risk profiles.

---

## 🛠 Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Bias Correction** | Random Forest | Estimating propensity scores to handle non-response bias. |
| **Main Classifier** | MLP (Neural Network) | Predicting adoption willingness (Binary Classification). |
| **Data Handling** | Pandas & NumPy | Synthetic feature engineering and weight calculation. |
| **NLP Simulation** | Keyword-based Logic | Generating sentiment features from qualitative feedback. |
| **Evaluation** | Scikit-Learn Metrics | Precision, Recall, and F1-Score analysis. |

---

## 🚀 Key Achievements

* **Bias Mitigation:** Implemented a robust correction for "Survey Non-response," a common pitfall in large-scale social and agricultural data analysis.
* **Feature Synergy:** Successfully bridged environmental data (soil/rainfall) with psychological/social data (trust scores).
* **Scalable Pipeline:** The architecture is designed to ingest raw CSV data and output a weighted, sentiment-aware prediction in a single, unified execution flow.

---

## 📊 Evaluation Results

The model provides a detailed `classification_report`, allowing researchers to see how well the model predicts "Adopters" vs. "Non-Adopters" while accounting for the recalculated statistical weights.

> **Note:** In this context, we prioritize the **F1-Score**, as it balances the need to identify potential adopters without over-predicting and wasting limited outreach resources.

---

## 🏁 Quick Start

### 1. Installation
```bash
git clone [https://github.com/your-username/agri-ml.git](https://github.com/your-username/agri-ml.git)
cd agri-ml
pip install pandas numpy scikit-learn
```
### 2. Run Pipeline

Execute the full ingestion, weighting, and prediction flow:
```bash
python main.py
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| F1-Score | X |
   
---

## 📉 Future Roadmap

* **[ ] Geospatial Clustering:** Integrate GPS coordinates to account for spatial autocorrelation in adoption patterns.
* **[ ] SHAP Explainability:** Implement SHAP (SHapley Additive exPlanations) to provide farmers and stakeholders with "Local Explanations" for predictions.
* **[ ] Time-Series Extension:** Transition to an LSTM architecture to analyze how propensity changes over multiple planting seasons.

