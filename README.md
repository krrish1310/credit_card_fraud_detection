# 💳 Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using anomaly detection methods (Isolation Forest, Local Outlier Factor) and supervised learning (XGBoost). The project includes a Streamlit-based web dashboard for real-time predictions.

---

## 📁 Project Structure


---

## 📌 Dataset

- Source: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Description: Contains transactions made by European cardholders in September 2013. The dataset is highly imbalanced and features are anonymized using PCA.

---

## 🧠 Machine Learning Workflow

1. **Preprocessing**:
   - Scaled `Amount` and `Time` features using `StandardScaler`
   - Balanced dataset via undersampling of legit transactions

2. **Anomaly Detection**:
   - `IsolationForest`
   - `LocalOutlierFactor`

3. **Supervised Learning**:
   - Trained `XGBoost` model
   - ROC Curve & Confusion Matrix for evaluation

4. **Model Export**:
   - Saved using `joblib` → `xgb_model.pkl`

5. **Web App**:
   - Built using Streamlit for interactive fraud prediction

---

## 🚀 How to Run

### 🔧 1. Install dependencies

```bash
pip install -r requirements.txt

