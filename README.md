# 🩺 Multiclass Sleep Health Risk Classification & Explainable AI System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-Classifier-blueviolet.svg)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
[![Decision Tree](https://img.shields.io/badge/Decision%20Tree-Classifier-success.svg)](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
[![Random Forest](https://img.shields.io/badge/Random%20Forest-Classifier-green.svg)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
[![XGBoost](https://img.shields.io/badge/XGBoost-Final%20Model-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![Support Vector Machine](https://img.shields.io/badge/Support%20Vector%20Machine-SVM-red.svg)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-9cf.svg)](https://shap.readthedocs.io/)
[![imbalanced--learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-red.svg)](https://imbalanced-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-Deployment-lightgrey.svg)](https://flask.palletsprojects.com/)

> A supervised, explainable machine learning system that classifies an individual's sleep health status — identifying whether a sleep disorder (**Insomnia** or **Sleep Apnea**) is present, or confirming a healthy sleep profile — using demographic, lifestyle, physiological data & cardiovascular health indicators. Built as a leakage-free, benchmarked, and interpretable end-to-end pipeline, deployed as a Flask web application.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Business & ML Objectives](#-business--ml-objectives)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering](#-feature-engineering)
- [Data Preprocessing](#-data-preprocessing)
- [Modeling Approach](#-modeling-approach)
- [Results & Model Comparison](#-results--model-comparison)
- [Model Explainability (SHAP)](#-model-explainability-shap)
- [Key Insights](#-key-insights)
- [Conclusion](#-conclusion)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Future Enhancements](#-future-enhancements)
- [Disclaimer](#-disclaimer)
- [Author](#-author)

---

## 🎯 Project Overview

Sleep disorders like **Insomnia** and **Sleep Apnea** are frequently under-diagnosed despite strong ties to measurable lifestyle and physiological signals — stress, activity level, blood pressure, heart rate. This project builds a full ML pipeline, from raw data to a deployed, interpretable model, that classifies sleep-disorder status using **10,000 patient records** across demographic, occupational, lifestyle, and cardiovascular features.

| Objective | Outcome |
|---|---|
| **Early Detection** | Flags at-risk individuals from routine health/lifestyle data |
| **Model Transparency** | SHAP explains *why* each prediction was made, not just *what* |
| **Deployment-Ready** | Packaged as a Flask web app for real-time inference |

---

## 🧩 Problem Statement

Given an individual's demographic, lifestyle, and physiological attributes, predict their sleep-disorder status as one of three classes:

- **No Sleep Disorder** — no clinical indicators present
- **Insomnia** — difficulty falling or staying asleep
- **Sleep Apnea** — disrupted breathing during sleep

---

## 💼 Business & ML Objectives

**Business Objective**
- Classify sleep-disorder status from routinely available health and lifestyle data.
- Benchmark class-imbalance strategies and select the strongest model on class-aware metrics.
- Explain predictions with **SHAP** for clinical trust and transparency.
- Deploy the final model as a **Flask** web application.

**ML Objective**
- Build, tune, and evaluate supervised multiclass models using **stratified cross-validation**.
- Select the final model on **generalization, class balance, and robustness** — not raw accuracy alone, given a 78/12/10 class split.

| Attribute | Description |
|---|---|
| **Learning Type** | Supervised Learning |
| **Problem Category** | Multiclass Classification |
| **Target Variable** | `Sleep Disorder` |
| **Target Classes** | No Sleep Disorder · Insomnia · Sleep Apnea |

---

## 📊 Dataset

**Source:** Sleep Health and Lifestyle Dataset (Kaggle, synthetically generated)  
**File:** `data/Sleep_Health_Lifestyle_Dataset.xlsx`  
**Raw Shape:** 10,050 records → **10,000 records × 13 features** after de-duplication

| Feature | Description |
|---|---|
| Gender, Age, Occupation | Demographic attributes |
| Sleep Duration, Quality of Sleep | Core sleep metrics |
| Physical Activity Level, Stress Level | Lifestyle indicators |
| BMI Category | Weight classification |
| Systolic BP, Diastolic BP | Cardiovascular indicators (split from combined `Blood Pressure` field) |
| Heart Rate, Daily Steps | Physiological & activity metrics |
| **Sleep Disorder** *(Target)* | No Sleep Disorder / Insomnia / Sleep Apnea |

**Data Quality Checks**
- ⚠️ **50 duplicate records** identified and removed.
- ⚠️ **12 of 13 features contain missing values** (up to ~5.4% per feature) — handled via median/mode imputation inside a leakage-free pipeline, not by dropping rows.
- ✅ `"None"` in the target column was explicitly preserved as a valid category (not read as null) using `keep_default_na=False` during ingestion, then relabeled to `"No Sleep Disorder"`.
- ✅ Categorical inconsistencies (`FEMALE`, `female`, `Female `) standardized before encoding.

**Target Distribution** — the dataset is class-imbalanced:

| Class | Count | % |
|---|---|---|
| No Sleep Disorder | 7,813 | 78.13% |
| Insomnia | 1,237 | 12.37% |
| Sleep Apnea | 950 | 9.50% |

---

## 🔄 Project Workflow

The notebook follows an 8-phase, leakage-aware pipeline:

1. **Initial Data Preparation** — deduplication, text/category standardization, type correction, BP splitting, target relabeling.
2. **Data Understanding** — shape, schema, missing-value profile, descriptive statistics, cardinality, target distribution.
3. **Exploratory Data Analysis** — univariate, categorical, and relationship analysis; correlation and outlier checks.
4. **Feature Engineering** — three domain-derived features from existing predictors (no target leakage).
5. **Data Preprocessing** — `ColumnTransformer`-based pipeline (impute → scale/encode), fit only on training data.
6. **Model Development & Selection** — 5 algorithms × 2 imbalance strategies, compared via stratified 5-fold CV.
7. **Hyperparameter Tuning** — `RandomizedSearchCV` on the top candidates, optimizing Macro F1.
8. **Final Evaluation & Explainability** — untouched test-set evaluation, confusion matrix, ROC-AUC, and SHAP analysis.

---

## 📈 Exploratory Data Analysis

Built with **Matplotlib** and **Seaborn** — distribution plots, categorical breakdowns, correlation heatmap, pair plots, and IQR-based outlier detection.

**Key Findings**
- **Sleep Apnea** patients show markedly higher **Systolic/Diastolic BP**, **Pulse Pressure**, and **Heart Rate** than the other classes.
- **Insomnia** is strongly associated with lower **Sleep Duration** and **Quality of Sleep**, and higher **Stress Level**.
- **BMI Category** shows the clearest categorical association with disorder status — **Obese** individuals skew heavily toward Sleep Apnea.
- **Occupation** (Doctor, Nurse, Lawyer) shows elevated disorder rates, likely as a stress/activity proxy; **Gender** shows minimal separation.
- Outliers are limited to **under 1%** of observations across all numerical features — flagged, not removed.

---

## 🛠 Feature Engineering

Three domain-relevant features were derived from existing predictors (target-independent):

| Feature | Formula | Rationale |
|---|---|---|
| **Pulse Pressure** | `Systolic BP − Diastolic BP` | Additional cardiovascular signal beyond raw BP |
| **Sleep Quality Index** | `((Sleep Duration / 24) + (Quality of Sleep / 10)) / 2` | Consolidated, standardized sleep-profile score |
| **Stress-Sleep Interaction** | `Stress Level × Sleep Duration` | Captures the combined effect of stress and sleep duration |

**Validation:** All three features separate classes meaningfully — Pulse Pressure is highest for Sleep Apnea (mean 66.5 vs. ~49 for the other two classes), and Sleep Quality Index is lowest for Insomnia (0.368 vs. ~0.50) — and were retained for modeling.

---

## ⚙️ Data Preprocessing

- **Split:** Stratified 80/20 train-test split → 8,000 train / 2,000 test.
- **Numerical pipeline:** median imputation → `StandardScaler`.
- **Categorical pipeline:** mode imputation → `OneHotEncoder`.
- Combined via `ColumnTransformer`, **fit only on training data** to prevent leakage — expands to a **30-feature** processed matrix.

---

## 🤖 Modeling Approach

Five classification algorithms were benchmarked under two imbalance-handling strategies — **class weighting** and **SMOTE** — using **Stratified 5-Fold Cross-Validation**. Preprocessing and SMOTE, where applicable, were refit independently inside every fold to prevent data leakage.

| Model | Imbalance Handling |
|---|---|
| Logistic Regression | Class Weight / SMOTE |
| Decision Tree | Class Weight / SMOTE |
| Random Forest | Class Weight / SMOTE |
| SVM | Class Weight / SMOTE |
| **XGBoost** | Class Weight / SMOTE |

**Primary metric:** **Macro F1**, which gives equal importance to each class and is appropriate for the imbalanced target distribution.

**Tuning:** `RandomizedSearchCV` was used for XGBoost and Random Forest, optimizing cross-validation Macro F1. The test set remained untouched until final evaluation.

---

## 🏆 Results & Model Comparison

### Cross-Validation Performance

| Model | Strategy | Best CV Macro F1 |
|---|---|---:|
| **XGBoost** | **Class Weight** | **85.33%** |
| XGBoost | SMOTE | 85.05% |
| Random Forest | Tuned | 83.26% |
| SVM | SMOTE | 81.35% |
| Logistic Regression | SMOTE | 81.00% |
| Decision Tree | Class Weight | 75.02% |

**XGBoost with Class Weight** achieved the highest cross-validation Macro F1 and was selected as the final model.

---

## 📈 Final Test-Set Performance

The final tuned **XGBoost** model was evaluated on a **held-out test set** that remained untouched during model training, cross-validation, and hyperparameter tuning.

| Metric | Score |
|---|---:|
| **Accuracy** | **93.50%** |
| **Macro F1** | **88.50%** |
| **Balanced Accuracy** | **87.43%** |
| **Macro Precision** | **89.65%** |
| **Macro Recall** | **87.43%** |
| **ROC-AUC** | **98.29%** |

### Class-wise Performance

| Class | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| No Sleep Disorder | **95.33%** | **96.67%** | **96.00%** |
| Sleep Apnea | **89.78%** | **87.89%** | **88.83%** |
| Insomnia | **83.84%** | **77.73%** | **80.67%** |

The final model achieved strong overall performance, with particularly strong classification of **No Sleep Disorder** and **Sleep Apnea**. **Insomnia** remained the most challenging class, with a lower recall and F1 score.

### Precision-Recall Performance

The **Precision-Recall (PR) curve** was used to evaluate model performance across classification thresholds, particularly for the imbalanced target classes.

| Class | PR-AUC |
|---|---:|
| No Sleep Disorder | **0.9929** |
| Sleep Apnea | **0.9585** |
| Insomnia | **0.8951** |

The PR-AUC results demonstrate strong precision-recall performance across all three classes, with **No Sleep Disorder** achieving the highest PR-AUC and **Insomnia** remaining the comparatively more challenging class.

**Final Model:** Tuned **XGBoost with Class Weight**

---

## 🔍 Model Explainability (SHAP)

SHAP (`TreeExplainer`) was applied to the final XGBoost model on the transformed test set to quantify feature contributions.

**Global Feature Importance (highest → lowest):**
1. **Heart Rate** — single strongest predictor across all classes.
2. **Sleep Duration** and **Sleep Quality Index** — next most influential.
3. **Systolic BP** and **Age** — substantial contribution.
4. **Physical Activity Level** and **Pulse Pressure** — meaningful supporting signal.
5. **Quality of Sleep** and **Stress-Sleep Interaction** — class-specific influence.
6. **BMI Category (Obese)** — noticeable, but secondary to physiological/sleep features.
7. **Occupation** features — lowest overall influence.

**Takeaway:** the model relies primarily on **physiological and sleep-related signals**, not demographic or occupational proxies — supporting its clinical plausibility and the use of SHAP for stakeholder trust, not just performance reporting.

---

## 💡 Key Insights

1. **Heart Rate, Sleep Duration, and Sleep Quality Index** are the strongest predictors of sleep-disorder status.
2. **Sleep Apnea** is driven by elevated **blood pressure and heart rate**; **Insomnia** by lower **sleep quality and duration** plus higher **stress**.
3. Class weighting and SMOTE perform comparably here — class weighting was marginally better and simpler to deploy (no synthetic data generation at inference time).
4. **XGBoost** outperformed all other candidates under every configuration tested, making it the deployment choice.
5. **Insomnia** remains the hardest class to separate from a healthy sleep profile — the clearest target for future feature or data improvements.

---

## 🏁 Conclusion

This project developed an **explainable multiclass machine learning solution** for classifying **No Sleep Disorder, Insomnia, and Sleep Apnea** using demographic, lifestyle, and physiological features.

After comparing multiple classifiers, class-weighting and SMOTE strategies, and tuning the strongest candidates, **XGBoost** achieved the best final test performance with **93.50% accuracy, 88.50% Macro F1, 87.43% balanced accuracy, and 98.29% ROC-AUC**. SHAP analysis provided insight into the features influencing model predictions, while class-level evaluation showed that **Insomnia was more difficult to distinguish from No Sleep Disorder** than the other classes. The final model is prepared for deployment through a **Flask-based web application**.

> **Note:** This project is intended for educational and predictive analytics purposes and is not a clinical diagnostic system.

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Preprocessing | Scikit-learn (`ColumnTransformer`, `Pipeline`, `StandardScaler`, `OneHotEncoder`) |
| Imbalance Handling | `imbalanced-learn` (SMOTE) |
| Modeling | Scikit-learn (Logistic Regression, Decision Tree, Random Forest, SVM), XGBoost |
| Tuning | `RandomizedSearchCV` (Stratified 5-Fold CV) |
| Explainability | SHAP |
| Deployment | Flask |
| Environment | Jupyter Notebook |

---

## 📁 Project Structure

```text
Multiclass-Sleep-Disorder-Classification/
│
├── app.py
├── model.py
├── sleep_model.pkl
├── requirements.txt
├── render.yaml
├── .gitignore
├── README.md
│
├── data/
│   ├── Sleep_Health_Lifestyle_Dataset.xlsx
│   ├── Multiclass Sleep Disorder Classification Report.pdf
│   └── Multiclass Seep Disorder Classification Report.docx
│
├── notebook/
│   └── Multiclass Sleep Disorder Classification.ipynb
│
├── templates/
│   └── index.html
│
├── figures/
│   ├── Box_Plot_Analysis_Numerical_Features.png
│   ├── Categorical_Feature_Distribution.png
│   ├── Categorical_Features_vs_Sleep_Disorder_Chart.png
│   ├── Classwise_Precision_Recall_F1_Score.png
│   ├── Correlation_Heatmap_Numerical_Features.png
│   ├── Cross_Validated_MacroF1_ClassWeight_vs_SMOTE.png
│   ├── MultiClass_ROC_AUC_Curve_XGBoost.png
│   ├── Numerical_Feature_Distribution.png
│   ├── Numerical_Features_vs_Sleep_Disorder_BoxPlot.png
│   ├── PairPlot_Scatter_Matrix_Distribution.png
│   ├── SHAP_Feature_Interaction_Summary_Plot.png
│   ├── SHAP_Global_Feature_Importance_Chart.png
│   ├── Sleep_Disorder_Distribution.png
│   └── XGBoost_Confusion_Matrix.png
│
├── dashboard/
│   ├── Sleep_Health_Analytics_Dataset.xlsx
│   ├── Sleep_Health_Analytics_Dashboard.pbix
│   ├── README.md
│   ├── DAX_Measures.md
│   └── screenshots/
│       ├── sleep-health-analytics-dashboard.png
│       └── sleep-health-data-model.png
│

```

---

## ▶️ How to Run & Render Deployment

### 💻 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/rushikesh-kh/Multiclass-Sleep-Disorder-Classification.git
cd Multiclass-Sleep-Disorder-Classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Explore the notebook
jupyter notebook notebook/Multiclass%20Sleep%20Disorder%20Classification.ipynb

# 4. Run the Flask application locally
python app.py
```

The application will be available locally at:

```text
http://127.0.0.1:5000
```

### 🌐 Render Deployment — Live Demo

The application is deployed as a Flask web service on **Render** using **Gunicorn**.

**Live Demo:**  
https://multiclass-sleep-disorder-classification.onrender.com

The deployed application provides the same sleep-disorder prediction functionality through a web interface without requiring local installation.

### ⚙️ Render Deployment Configuration

The deployment is configured using the `render.yaml` file included in the repository:

```yaml
services:
  - type: web
    name: sleep-disorder-prediction
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    healthCheckPath: /health
```

Render automatically:

1. Connects to the GitHub repository.
2. Installs the dependencies from `requirements.txt`.
3. Starts the Flask application using Gunicorn.
4. Runs the `/health` endpoint to verify that the service is running.
5. Provides a public URL for accessing the deployed application.

### ❤️ Health Check

The application provides a health-check endpoint:

```text
https://multiclass-sleep-disorder-classification.onrender.com/health
```

A successful response is:

```json
{
  "status": "ok"
}
```

This confirms that the Flask application is running successfully.

### 📦 Production Deployment

The deployment architecture is:

```text
GitHub Repository
       ↓
     Render
       ↓
Install requirements.txt
       ↓
    Gunicorn
       ↓
 Flask Application
       ↓
  Live Web App
```

### ⚠️ Render Free Instance

The application is deployed using Render's Free instance. Free instances may spin down after periods of inactivity, so the first request after a period of inactivity may take a little longer while the service starts.

### 📋 Requirements

The project dependencies are defined in `requirements.txt`:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
shap
flask
openpyxl
```
---

## 🚀 Future Enhancements

- Validate on real clinical data — the current dataset is synthetic.
- Add a personalized sleep-risk score with lifestyle recommendations, moving from classification to decision support.
- Incorporate wearable-device time-series data (e.g., sleep-stage tracking) for finer-grained prediction.
- Extend the Flask app with SHAP-based per-prediction explanations in the UI.

---

## ⚠️ Disclaimer

This project uses a **synthetic dataset** and is built for educational and portfolio purposes only. Predictions are **not a medical diagnosis** and should never replace professional healthcare advice. Anyone with sleep-related health concerns should consult a qualified healthcare provider.

---

## 👤 Author

**Rushikesh Khamgaonkar**  
📧 [rushikeshkhamgaonkar9869@gmail.com](mailto:rushikeshkhamgaonkar9869@gmail.com) · [LinkedIn](https://www.linkedin.com/in/rushikesh-khamgaonkar-588b77227/) · [GitHub](https://github.com/rushikesh-kh)

<p align="center"><i>⭐ If you found this project insightful, consider starring the repository!</i></p>
