# Quantitative Risk & Machine Learning Portfolio

> Data science and machine learning applied to credit-risk decisions — consumer approval, bank default, and corporate default — using both machine-learning and classical statistical methods.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-3B5998?style=flat)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-7C3AED?style=flat)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)

---

## Overview

Three end-to-end projects on credit-risk decisions, from approving a card applicant to estimating a company's Probability of Default (PD). Each project frames the business problem, prepares and explores the data, builds and tunes models, and validates them with risk-relevant metrics.

Two methodologies are covered on purpose: modern machine learning and classical statistical / econometric modeling.

---

## Key Skills

| Category | Tools & Techniques |
| --- | --- |
| **Language** | Python |
| **Data Handling** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Classical ML** | scikit-learn — Logistic Regression, SVM, Decision Trees, Random Forests |
| **Statistical & Econometric Modeling** | statsmodels — Logit / Probit / OLS (Linear Probability Model), cluster-robust & HC1 robust standard errors, marginal effects, quasi-separation diagnostics |
| **Gradient Boosting** | XGBoost — native missing-value handling, early stopping |
| **Deep Learning** | TensorFlow, Keras (feed-forward neural networks) |
| **Explainable AI** | SHAP — global & local interpretation of black-box models |
| **Credit-Risk Toolkit** | scorecardpy — Weight of Evidence (WoE), Information Value (IV), deployable points scorecard; expert-scorecard benchmarking |
| **Imbalanced Data** | imbalanced-learn — SMOTE, SMOTETomek, undersampling |
| **Feature Selection** | IV filtering, correlation analysis, VIF & multicollinearity control |
| **Validation & Stability** | Stratified, grouped (`StratifiedGroupKFold`), and out-of-time (crisis-window) validation; Population Stability Index (PSI); bootstrap confidence intervals; calibration / reliability curves; segmentation testing |
| **ML Workflow** | Preprocessing pipelines, data-leakage prevention, hyperparameter tuning (`RandomizedSearchCV`), cross-validation, reproducible (pinned) environments |
| **Model Evaluation** | AUC-ROC, Gini, KS statistic, Brier score, Precision / Recall / F1, confusion matrices, decision-threshold optimization |
| **Environment** | Jupyter Notebook, Google Colab |

---

## Projects

### 1. Bank Probability-of-Default (PD) — Scorecard, XGBoost & Explainable AI

**Problem.** Estimate each borrower's Probability of Default (PD) to rank, price, and provision for credit risk. In a regulated setting the model must be accurate, calibrated, stable over time, and explainable enough for model-risk governance. Data: ~35,000 bank-year records, 26 financial-ratio features, ~2.8% default rate.

**Approach.** A full credit-risk pipeline (Phases 0–14): EDA of default rates by year and region, data-quality treatment, Weight-of-Evidence (WoE) transformation, and variable selection by Information Value, correlation, and VIF. Two models are compared — an interpretable logistic / probit scorecard and an XGBoost challenger (tuned with `RandomizedSearchCV` and early stopping) — interpreted with SHAP, tested out-of-time on the 2007–2009 crisis, and checked for stability with PSI. All transformations are fit on training data only; fixed seed.

**Tech.** `Python` · `pandas` · `NumPy` · `scikit-learn` · `statsmodels` · `scorecardpy` · `XGBoost` · `SHAP` · `SciPy` · `Matplotlib` · `Seaborn`

**Results.**
- XGBoost gave the best discrimination (test AUC ≈ 0.86, Gini ≈ 0.72). The logistic and probit scorecards were close (AUC ≈ 0.84); a shallow decision tree was weakest (≈ 0.70).
- The scorecard generalized best (smallest train–test gap). XGBoost was more accurate but overfit more.
- Both models stayed calibrated to the actual default rate across regions and years, and kept discriminatory power through the 2007–2009 crisis when trained only on pre-crisis data.
- PSI stayed below the 0.10 stability threshold, so the scored population matched the training population.
- The final logistic model was converted to a points scorecard; both models scored a fresh applicant sample.
- Recommendation: use the WoE + logistic scorecard as the primary regulated model, XGBoost as a challenger, and SHAP for explainability.

**Notebook:** [`credit_risk_explainable_ai.ipynb`](./credit_risk_explainable_ai.ipynb)

---

### 2. Corporate Probability-of-Default (PD) — Econometric Models, Expert Benchmark & Robust Inference

**Problem.** Estimate a company's PD from its history and expert financial assessments (profitability, liquidity, credit access, management quality). Priority: a statistically rigorous, interpretable, defensible model, benchmarked against the firm's existing expert scorecard.

**Approach.** A leakage-aware PD pipeline focused on statistical rigor. Customer-history features are built with no look-ahead; the panel is split with `StratifiedGroupKFold` so no customer's records cross train and test. After train-only WoE and IV selection, three models are compared — logistic, probit, and a linear probability model (LPM) — with cluster-robust standard errors, marginal effects, multicollinearity checks, and bootstrap confidence intervals. Models are benchmarked against the expert scorecard, with a segmentation test for per-group models.

**Tech.** `Python` · `pandas` · `NumPy` · `statsmodels` · `scikit-learn` · `scorecardpy` · `SciPy` · `Matplotlib` · `Seaborn`

**Results.**
- Logistic and probit were statistically indistinguishable (the bootstrap CI for their Gini difference contained zero). Logistic was chosen for interpretability and regulatory familiarity.
- The linear probability model was rejected: about 42% of its predicted probabilities fell outside [0, 1].
- Benchmarking found an orientation bug in the expert scorecard — it ranked the healthiest firms as the riskiest. After correction it discriminated almost as well as the fitted model but stayed badly mis-calibrated: good at ranking risk, weak at estimating the level of PD.
- Segmentation (with paired bootstrap CIs) showed one overall model was enough; per-segment models added no discriminatory power.
- Stated limitations: the expert-assessment features partly encode the credit decision, which inflates discrimination; production use would need out-of-time and PSI testing.

**Notebook:** [`credit_risk_pd.ipynb`](./credit_risk_pd.ipynb)

---

### 3. Credit Card Approval Prediction

**Problem.** Predict which card applicants to approve — a consumer credit-risk problem. Approving a future defaulter causes losses; rejecting a creditworthy applicant loses revenue.

**Approach.** A standard ML pipeline: EDA, one-hot encoding, feature standardization, and correction of severe class imbalance (~0.5% rejections) with SMOTETomek. Five models are trained and compared, each tuned by cross-validated search. Train and test sets are preprocessed separately to prevent leakage.

**Tech.** `Python` · `pandas` · `NumPy` · `scikit-learn` · `imbalanced-learn` · `TensorFlow / Keras` · `Matplotlib` · `Seaborn`

**Results.**
- The data was highly imbalanced (~99.5% approvals), so resampling and class weighting were needed for the models to detect high-risk applicants.
- Logistic Regression and SVM performed best: on the test set the logistic model identified the reject cases while keeping false approvals near zero.
- Decision Tree and Random Forest also worked but were more sensitive to hyperparameters.
- A tuned neural network did not beat the simpler models.
- Takeaway: preprocessing (imbalance handling and feature scaling) mattered more than model complexity.

**Notebook:** [`credit_card_project.ipynb`](./credit_card_project.ipynb)

---

## How to Navigate

Each project is a self-contained Jupyter notebook (`.ipynb`).

- **View in browser:** click any `.ipynb` above; GitHub renders notebooks directly, so you can read the code, charts, and explanations without installing anything.
- **Run it:** download the notebook or open it in [Google Colab](https://colab.research.google.com/).

Each notebook reads top to bottom: problem → data exploration → preprocessing → modeling → validation → conclusions.

---

## Contact

Open to opportunities in quantitative risk and data science.

- **Email:** [karol.kudlacik@outlook.com](mailto:karol.kudlacik@outlook.com)
- **LinkedIn:** [karol-kudlacik](https://www.linkedin.com/in/karol-kudlacik)
- **GitHub:** [karolkudlacik](https://github.com/karolkudlacik)
