# 🏦 Bankruptcy Prediction Hackathon

This repository contains the complete workflow for predicting corporate bankruptcy using financial indicators.  
Developed for a machine learning hackathon, organized by **Department of Management Studies (DMS), Indian Institutes of Technology (IIT) (ISM), Dhanbad**. This project focuses on maximizing **Macro-F1 Score** (primary metric) and **Recall on the “failed” class** (tie-breaker metric).

#### About the hackathon: 
 - Name: [*Predict2Protect*](https://unstop.com/college-fests/plutus-indian-institute-of-technology-indian-school-of-mines-iit-ism-dhanbad-409223)
 - Predict2Protect is a week-long data-driven hackathon under PLUTUS’25, where participants build machine learning models to predict corporate bankruptcy using real-world financial datasets.
 - The competition aims to evaluate end-to-end data science skills — from data preprocessing and feature engineering to model optimization and business interpretation.
---

## 📊 Project Overview

Financial distress prediction is a crucial task in corporate finance and risk management.  
Given anonymized financial data of companies over multiple years, the goal is to classify each company as either:

- **`alive`** — company remains solvent  
- **`failed`** — company declared bankruptcy  

---

## 🧠 Modeling Stages

### ✅ Stage 1 — Random Forest + XGBoost (Baseline)
**Goal:** quick benchmark + imbalance handling trial  
**Techniques**
- Feature engineering on financial ratios
- SMOTE + class weights
- Stratified train/val split

**Result**
- Macro-F1 (LB): ~0.50  
- Good baseline but struggled with minority recall

---

### ✅ Stage 2 — CatBoost + Optuna (Leak-corrected)
**Why:** Tree models exploit categorical structure well; tuned aggressively

**Key Steps**
- Removed leakage features (MajorGroup, etc.)
- CatBoost with supervised encoding
- Optuna hyperparameter tuning
- Threshold tuning for macro-F1

**Result**
- Macro-F1 improved to ~0.5077 on leaderboard
- Validation macro-F1 ~0.57 after leak fix

---

### ✅ Stage 3 — Weighted Ensemble (CatBoost + LightGBM + XGBoost)
**Why:** Single models plateaued — combine strengths  
**Method**
- Ordinal encoding (CatBoost raw cat)
- Train CB + LGBM + XGB with early stopping
- Optimize ensemble weights + decision threshold using Optuna

**Final Weights**
| Model    | Weight |
| -------- | ------ |
| LightGBM | 0.614  |
| XGBoost  | 0.211  |
| CatBoost | 0.175  |

**Threshold:** ~0.21  
**Val Macro-F1:** ~0.656

📌 **Leaderboard improvement:** 0.5022 → ~0.51+ (post-ensemble we expect stronger jump)

---

## 🧪 Techniques Used
- Advanced tree boosting (CatBoost, LightGBM, XGBoost)
- Imbalanced classification strategies:
  - threshold optimization
  - weighted loss
  - stratified splits
- Leakage detection + elimination
- Optuna Bayesian hyperparameter search
- Model stacking via weighted average ensemble

---

## 📂 Repository Structure

```
📁 /models
├── 01_rfc_xgboost.ipynb
├── 02_catboost_optuna.ipynb
└── 03_weighted_ensemble.ipynb (current best)
📁 /data 
📄 README.md
📄 requirements.txt
```

---

## 🧠 Key Takeaways
- Imbalance + financial ratios = threshold optimization > plain models
- Removing leakage dropped score but made model real
- Ensembles beat individual models consistently
- Optuna saves time vs manual grid search

---

## 🚀 Next Steps
- Time-series / temporal split to mimic real bankruptcy prediction
- Pseudo-labeling on test data
- Winsorization / financial-domain ratio transformations
- SHAP-based risk factor explanations

---

## 👤 Author
**Sukrat Singh — IIT (ISM) Dhanbad**  
