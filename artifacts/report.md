# Bankruptcy Risk Modeling – Training Pipeline Report
*Generated:* 2025-08-13 21:17

## EDA (Jot Notes)
- Numeric financial ratios only → no encoding needed.
- Strong class imbalance (~3% bankrupt) → plan class weighting & stratification.
- Detected highly correlated ratios → apply correlation filtering (|r|>0.95).
- Outliers present; keep unless clear errors, as they may signal distress.

## Data Preprocessing (Jot Notes)
- Median imputation for missing values (robust to outliers).
- Standardize features for LR & NB; RF uses raw numeric features.
- Imbalance handled with `class_weight='balanced'` for LR & RF.
- Stratified train/test split; PSI checks for sampling bias.

## Feature Selection (Jot Notes)
- Simple correlation filter: drop one of any pair with |r|>0.95.
- Keep the feature with stronger correlation to target.
- Reduces multicollinearity for LR; minimal impact on RF.
- Selected 75 features (dropped 20).

## Hyperparameter Tuning (Jot Notes)
- Keep-it-simple: Grid for LR (C), RandomizedSearch for RF, light grid for NB.
- CV: Stratified 5-fold; score: ROC-AUC.
- Balance compute and performance (n_iter=20 for RF).
- Save best params for reproducible training.

## Model Training (Jot Notes)
- Models: Logistic Regression (benchmark), Random Forest, GaussianNB.
- Use CV during tuning; refit best estimator on full train.
- Save fitted models & scaler to disk for deployment.
- Consistent pipeline across models for fair comparison.

## Model Evaluation & Comparison (Jot Notes)
- Compare ROC-AUC, PR-AUC, F1, Precision, Recall, Brier on train & test.
- Plot ROC, PR, and Calibration with train/test overlay per model.
- Watch AUC gaps & calibration for over/underfitting.
- Select best by test ROC-AUC (tie-breaker: PR-AUC & Brier).

## SHAP Interpretability (Jot Notes)
- Compute SHAP for the best test model (TreeExplainer for RF).
- Provide beeswarm/bar plots (top drivers of risk).
- Align top features with finance intuition (leverage, profitability, liquidity).
- Supports regulatory explainability & risk review.

## PSI / Drift (Jot Notes)
- PSI(train vs test) per feature; high PSI suggests sampling bias.
- Review top drifted features + overlay distributions.
- If PSI > 0.25 repeatedly → re-sample or re-split; monitor in prod.
- Adds guardrail for stable deployment.

## Challenges & Reflections (Jot Notes)
- Heavy class imbalance → precision/recall trade-offs by model.
- Many correlated ratios → needed simple filtering to stabilize LR.
- Calibration varies; Brier score/curves reveal miscalibration.
- SHAP on trees can be slow; sub-sampled to keep runtime reasonable.

## Selected Features
- Count: 75
- Dropped: 20

### Metrics Summary (Train/Test)

| model               |   roc_auc_train |   roc_auc_test |   pr_auc_train |   pr_auc_test |   brier_train |   brier_test |   f1_train |   f1_test |   precision_train |   precision_test |   recall_train |   recall_test |
|:--------------------|----------------:|---------------:|---------------:|--------------:|--------------:|-------------:|-----------:|----------:|------------------:|-----------------:|---------------:|--------------:|
| Logistic Regression |        0.954714 |       0.929322 |       0.443282 |      0.369055 |     0.0917873 |    0.0950217 |  0.314651  | 0.297521  |         0.19025   |        0.181818  |       0.909091 |      0.818182 |
| Random Forest       |        0.992022 |       0.945541 |       0.765627 |      0.422238 |     0.023993  |    0.0344183 |  0.67433   | 0.466667  |         0.508671  |        0.368421  |       1        |      0.636364 |
| GaussianNB          |        0.89376  |       0.802738 |       0.24649  |      0.168495 |     0.663173  |    0.640968  |  0.0842572 | 0.0796731 |         0.0440381 |        0.0417112 |       0.971591 |      0.886364 |

### Best Model
{
  "best_model": "Random Forest",
  "test_roc_auc": 0.9455,
  "model_path": "models\\Random_Forest.joblib",
  "prep": {
    "imputer": "prep/imputer.joblib",
    "scaler": "prep/scaler.joblib",
    "selected_features_csv": "prep/selected_features.csv"
  },
  "shap_assets": {
    "skipped": "shap/shap_skipped.txt"
  },
  "saved_models": {
    "Logistic Regression": {
      "path": "models\\Logistic_Regression.joblib",
      "best_params": {
        "C": 0.1
      }
    },
    "Random Forest": {
      "path": "models\\Random_Forest.joblib",
      "best_params": {
        "n_estimators": 200,
        "min_samples_split": 10,
        "max_features": "sqrt",
        "max_depth": 8
      }
    },
    "GaussianNB": {
      "path": "models\\GaussianNB.joblib",
      "best_params": {
        "var_smoothing": 1e-06
      }
    }
  }
}

### Key Artifacts
- EDA describe: `eda/feature_describe.csv`
- EDA missing: `eda/missing_values.csv`
- EDA correlation heatmap: `eda/correlation_heatmap_subset.png`
- Class balance: `eda/class_balance.png`
- PSI scores: `psi/psi_scores.csv`
- PSI topN: `psi/psi_topn.png`

- Logistic Regression ROC: `plots\Logistic_Regression\roc.png`
- Logistic Regression PR: `plots\Logistic_Regression\pr.png`
- Logistic Regression Calibration: `plots\Logistic_Regression\calibration.png`
- Random Forest ROC: `plots\Random_Forest\roc.png`
- Random Forest PR: `plots\Random_Forest\pr.png`
- Random Forest Calibration: `plots\Random_Forest\calibration.png`
- GaussianNB ROC: `plots\GaussianNB\roc.png`
- GaussianNB PR: `plots\GaussianNB\pr.png`
- GaussianNB Calibration: `plots\GaussianNB\calibration.png`
