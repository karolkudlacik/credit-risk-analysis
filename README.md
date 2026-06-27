# 📊 Quantitative Risk & Machine Learning Portfolio

> Applying data science and machine learning to real-world **credit-risk and financial decision problems** — spanning consumer approval, bank default, and corporate default modeling, with both machine-learning and classical-statistics approaches.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-3B5998?style=flat)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-7C3AED?style=flat)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

![Domain](https://img.shields.io/badge/Domain-Credit_Risk_%2F_Fintech-2E7D32?style=flat)
![Focus](https://img.shields.io/badge/Focus-Quant_%26_ML-1565C0?style=flat)
![Status](https://img.shields.io/badge/Portfolio-Complete-2E7D32?style=flat)

---

## 👋 Overview

Welcome! This repository is a portfolio of **data science and machine learning projects focused on quantitative and risk problems in financial services**.

Each project takes a real financial decision — from approving a credit applicant to estimating a company's probability of default — and works through it end to end: framing the business and risk objectives in plain terms, preparing and exploring the data, building and tuning models, and validating them with the metrics that actually matter for risk-based decision-making.

The goal is to show not just that I can build models, but that I can **connect technical work to business value**: managing risk, avoiding costly mistakes, building models that hold up over time, and supporting smarter decisions with data.

> 🛈 *Three complete, end-to-end projects, deliberately spanning two methodologies — modern **machine learning** (Project 2) and classical **statistical / econometric** modeling (Project 3) — applied to the same family of credit-risk problems.*

---

## 🛠️ Key Skills

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
| **Validation & Stability** | Stratified, **grouped (`StratifiedGroupKFold`)**, and **out-of-time (crisis-window) validation**; Population Stability Index (PSI); bootstrap confidence intervals; calibration / reliability curves; segmentation testing |
| **ML Workflow** | Preprocessing pipelines, **data-leakage prevention**, hyperparameter tuning (`RandomizedSearchCV`), cross-validation, reproducible (pinned) environments |
| **Model Evaluation** | AUC-ROC, Gini, KS statistic, Brier score, Precision / Recall / F1, confusion matrices, decision-threshold optimization |
| **Environment** | Jupyter Notebook, Google Colab |

---

## 📁 Projects

### 🏦 1. Bank Probability-of-Default (PD) Modeling — Scorecard, XGBoost & Explainable AI

**📌 Objective — the business problem**
Banks and their regulators need to estimate each borrower's **Probability of Default (PD)** — the chance it fails to repay — in order to rank, price, and provision for credit risk. In a regulated setting a PD model can't be a pure black box: it must be **accurate, well-calibrated, stable over time, and explainable** enough to satisfy internal model-risk governance and supervisors. This project builds exactly that on a panel of **~35,000 bank-year observations** (26 financial-ratio features, ~2.8% default rate), delivering both a transparent regulatory **scorecard** and a high-performance **challenger** model.

**⚙️ Approach**
A complete, audit-ready credit-risk pipeline covering the full modeling lifecycle (Phases 0–14): exploratory analysis of default rates by year and region, data-quality treatment, **Weight-of-Evidence (WoE)** transformation, and disciplined variable selection via **Information Value, correlation, and VIF**. Two paradigms were then built and compared — an interpretable **logistic / probit scorecard** and an **XGBoost** challenger (tuned with `RandomizedSearchCV` + early stopping) — interpreted with **SHAP**, stress-tested on the **2007–2009 financial crisis** as an out-of-time validation, and checked for **population stability (PSI)**. Every transformation is fit on the training set only, with a fixed seed for full reproducibility.

**💻 Tech Stack**
`Python` · `pandas` · `NumPy` · `scikit-learn` · `statsmodels` · `scorecardpy` · `XGBoost` · `SHAP` · `SciPy` · `Matplotlib` · `Seaborn`

**📈 Key Results & Insights**
- **XGBoost achieved the strongest discrimination** (test AUC ≈ 0.86 / Gini ≈ 0.72), with the **logistic & probit scorecards close behind** (AUC ≈ 0.84) and a shallow decision tree the weakest (≈ 0.70) — a clear illustration of the **interpretability ↔ performance trade-off**.
- The **scorecard generalized best** (smallest train–test gap), while XGBoost, though more accurate, showed more overfitting — a key consideration for production stability.
- Both models stayed **well-calibrated** against the actual default rate across regions and years, and **retained discriminatory power through the 2007–2009 crisis** even when trained only on pre-crisis data — directly answering the question a model-risk validator cares about most.
- The **Population Stability Index stayed below the 0.10 "stable" threshold**, confirming the model scores a population consistent with the one it was built on.
- The final logistic model was converted into a **deployable points scorecard**, and both models were used to score a fresh sample of applicants — demonstrating a full path from raw data to a production-style scoring tool.
- **Recommendation:** for a regulated PD model, the transparent WoE + logistic scorecard is the defensible primary model, with XGBoost as a challenger/benchmark and SHAP supplying explainability.

**📓 Notebook:** [`credit_risk_explainable_ai.ipynb`](./credit_risk_explainable_ai.ipynb)

---

### 🏢 2. Corporate Probability-of-Default (PD) Modeling — Econometric Models, Expert Benchmark & Robust Inference

**📌 Objective — the business problem**
Lenders assessing **corporate** borrowers estimate each company's **Probability of Default (PD)** by combining its history with expert financial assessments (profitability, liquidity, access to credit, management quality, and so on). The emphasis here is on a model that is **statistically rigorous, interpretable, and defensible** — the qualities a credit-risk function and its regulators demand — and on honestly benchmarking it against the firm's existing **expert scorecard**.

**⚙️ Approach**
An end-to-end, leakage-aware PD pipeline built for **statistical rigor over raw predictive power**. Backward-looking customer-history features are engineered with no look-ahead, and the panel is split with **`StratifiedGroupKFold`** so that no single customer's records leak across the train and test sets. After train-only WoE transformation and IV-based selection, three models are compared — **logistic, probit, and a linear probability model (LPM)** — using **cluster-robust standard errors** (to handle repeated firm-years), marginal effects, multicollinearity / stability checks, and **bootstrap confidence intervals**. The models are then benchmarked against the firm's **expert scorecard**, with a segmentation analysis testing whether bespoke per-group models add value.

**💻 Tech Stack**
`Python` · `pandas` · `NumPy` · `statsmodels` · `scikit-learn` · `scorecardpy` · `SciPy` · `Matplotlib` · `Seaborn`

**📈 Key Results & Insights**
- **Logistic and probit regression were statistically indistinguishable** — the bootstrap confidence interval for their Gini difference contained zero — so logistic regression was chosen as the final model for its interpretability and regulatory familiarity.
- The **Linear Probability Model was rejected**: roughly **42% of its predicted "probabilities" fell outside the valid [0, 1] range**, a textbook demonstration of why a linear model is unsuitable for PD estimation.
- Benchmarking surfaced an **orientation bug in the expert scorecard** (it was ranking the healthiest firms as the riskiest). Once corrected, the expert score discriminated almost as well as the fitted model but **remained badly mis-calibrated** — strong at *ranking* risk, weak at estimating the *level* of PD: a clear, practical illustration of the difference between discrimination and calibration.
- A **segmentation analysis** (with paired bootstrap confidence intervals) showed a **single overall model was sufficient** — bespoke per-segment models added no discriminatory power.
- The notebook is candid about its **limitations** (e.g., the expert-assessment features partly encode the credit decision, inflating discrimination; out-of-time and PSI testing would be required for production) — the kind of critical, honest self-assessment expected in model-risk work.

**📓 Notebook:** [`credit_risk_pd.ipynb`](./credit_risk_pd.ipynb)

---

### 💳 3. Credit Card Approval Prediction

**📌 Objective — the business problem**
Lenders have to decide which applicants to approve for a credit card — a classic **consumer credit-risk** problem. Approving an applicant who later defaults exposes the lender to losses, while rejecting a creditworthy applicant means giving up revenue. This project builds a classification model that predicts the approval decision, helping the business balance **risk mitigation** (avoiding bad debt) against **profit maximization** (keeping good customers).

**⚙️ Approach**
Implemented a complete machine-learning pipeline: exploring and visualizing the applicant data, one-hot encoding categorical fields, standardizing numeric features, and correcting a severe class imbalance (only ~0.5% of records were rejections) using **SMOTETomek** resampling. Five models were then trained and compared, each tuned through cross-validated hyperparameter search. Crucially, the training and test sets were preprocessed **separately to prevent data leakage**, keeping the evaluation honest.

**💻 Tech Stack**
`Python` · `pandas` · `NumPy` · `scikit-learn` · `imbalanced-learn` · `TensorFlow / Keras` · `Matplotlib` · `Seaborn`

**📈 Key Results & Insights**
- The dataset was **highly imbalanced (~99.5% approvals)**, so resampling and class weighting were essential for the models to learn to recognize high-risk applicants at all.
- **Logistic Regression and SVM delivered the strongest results** — on the held-out test set the logistic model correctly identified the high-risk (reject) applicants while keeping false approvals at essentially zero, exactly the behavior a lender wants from a risk model.
- **Tree-based models (Decision Tree, Random Forest)** also performed well but were more sensitive to hyperparameter choices.
- A **neural network** was built and extensively tuned, but did not outperform the simpler models — a practical reminder that more complex models aren't always better for structured, tabular data.
- **Main takeaway:** disciplined data preprocessing (handling imbalance + feature scaling) had a larger impact on final performance than model complexity.

**📓 Notebook:** [`credit_card_project.ipynb`](./credit_card_project.ipynb)

---

## 🧭 How to Navigate This Repository

Each project lives in its own self-contained Jupyter Notebook (`.ipynb`).

- **🌐 View in your browser (no setup needed):** Just click any `.ipynb` file listed above. GitHub renders notebooks directly in the browser, so you can read all the code, charts, and explanations without installing anything.
- **▶️ Run it yourself:** Download the notebook or open it in [Google Colab](https://colab.research.google.com/) and run the cells interactively to reproduce the analysis.

Each notebook is organized to read top-to-bottom: problem framing → data exploration → preprocessing → modeling → validation → conclusions.

---

## 🤝 Let's Connect

I'm always open to discussing data science, quantitative risk, machine learning, or new opportunities. Feel free to reach out:

- 📧 **Email:** [karol.kudlacik@outlook.com]
- 💼 **LinkedIn:** [www.linkedin.com/in/karol-kudlacik]
- 🌐 **Portfolio / Website:** [https://github.com/karolkudlacik/karolkudlacik]

