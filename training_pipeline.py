import os
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
    brier_score_loss, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Fix Matplotlib Backend to Avoid Tkinter Warnings
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts

# Utility: ensure output dirs

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

# EDA: hist, box, correlation heat

def eda_plots(X: pd.DataFrame, y: pd.Series, out_dir: str, max_hist=24):
    eda_dir = ensure_dir(os.path.join(out_dir, "eda"))
    # Basic summary
    X.describe().to_csv(os.path.join(eda_dir, "feature_describe.csv"))
    X.isna().sum().sort_values(ascending=False).to_csv(os.path.join(eda_dir, "missing_values.csv"))

    # Histograms (cap to first N to avoid too many files)
    cols = X.columns[:max_hist]
    for c in cols:
        fig = plt.figure(figsize=(6,4))
        plt.hist(X[c].dropna().values, bins=50)
        plt.title(f"Histogram: {c}")
        plt.xlabel(c); plt.ylabel("Count")
        fig.tight_layout()
        fig.savefig(os.path.join(eda_dir, f"hist_{safe_name(c)}.png"))
        plt.close(fig)

    # Boxplots (same subset)
    for c in cols:
        fig = plt.figure(figsize=(6,4))
        plt.boxplot(X[c].dropna().values, vert=True)
        plt.title(f"Boxplot: {c}")
        plt.ylabel(c)
        fig.tight_layout()
        fig.savefig(os.path.join(eda_dir, f"box_{safe_name(c)}.png"))
        plt.close(fig)

    # Correlation heatmap (use smaller set to keep figure legible)
    corr = X[cols].corr().values
    fig = plt.figure(figsize=(8,6))
    im = plt.imshow(corr, aspect='auto', interpolation='nearest')
    plt.colorbar(im)
    plt.title("Correlation heatmap (subset)")
    plt.xticks(ticks=np.arange(len(cols)), labels=[short_name(c) for c in cols], rotation=90)
    plt.yticks(ticks=np.arange(len(cols)), labels=[short_name(c) for c in cols])
    fig.tight_layout()
    fig.savefig(os.path.join(eda_dir, "correlation_heatmap_subset.png"))
    plt.close(fig)

    # Class balance
    fig = plt.figure(figsize=(5,4))
    counts = y.value_counts().sort_index()
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Class balance (0=non-bankrupt, 1=bankrupt)")
    plt.xlabel("Class"); plt.ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(eda_dir, "class_balance.png"))
    plt.close(fig)

    return {
        "describe_csv": "eda/feature_describe.csv",
        "missing_csv": "eda/missing_values.csv",
        "histograms": [f"eda/hist_{safe_name(c)}.png" for c in cols],
        "boxplots": [f"eda/box_{safe_name(c)}.png" for c in cols],
        "corr_heatmap": "eda/correlation_heatmap_subset.png",
        "class_balance": "eda/class_balance.png"
    }

def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s)[:100]

def short_name(s: str, maxlen=16) -> str:
    s2 = s.strip()
    return (s2[:maxlen-1] + "…") if len(s2) > maxlen else s2

# Feature selection: simple corr filtering

def correlation_filter(X: pd.DataFrame, y: pd.Series, threshold: float = 0.95):
    """
    Remove one feature from pairs with |corr| > threshold.
    Keep the feature with higher absolute correlation to the target (point-biserial).
    """
    # correlation to target (binary 0/1)
    target_corr = X.apply(lambda col: pd.Series(col).corr(y), axis=0).abs().fillna(0.0)

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()

    for col in upper.columns:
        # find cols highly correlated with 'col'
        high = [row for row in upper.index if (upper.loc[row, col] > threshold)]
        for row in high:
            if row in to_drop or col in to_drop:
                continue
            # Drop the one with lower |corr| to target
            keep = col if target_corr[col] >= target_corr[row] else row
            drop = row if keep == col else col
            to_drop.add(drop)

    selected_cols = [c for c in X.columns if c not in to_drop]
    return selected_cols, sorted(list(to_drop))

# PSI (train vs. test)

def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index over fixed quantile buckets on expected (train)."""
    expected = np.array(expected).astype(float)
    actual = np.array(actual).astype(float)

    quantiles = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    # Avoid identical edges
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf

    expected_bins = np.digitize(expected, quantiles[1:-1])
    actual_bins = np.digitize(actual, quantiles[1:-1])

    exp_counts = np.bincount(expected_bins, minlength=buckets) / len(expected)
    act_counts = np.bincount(actual_bins, minlength=buckets) / len(actual)

    # Replace zeros to avoid div by zero / log issues
    exp_counts = np.where(exp_counts == 0, 1e-6, exp_counts)
    act_counts = np.where(act_counts == 0, 1e-6, act_counts)

    psi_vals = (exp_counts - act_counts) * np.log(exp_counts / act_counts)
    return float(np.sum(psi_vals))

def psi_report(X_train: pd.DataFrame, X_test: pd.DataFrame, out_dir: str, topn: int = 10):
    psi_dir = ensure_dir(os.path.join(out_dir, "psi"))
    psi_scores = []
    for col in X_train.columns:
        try:
            psi = calculate_psi(X_train[col].values, X_test[col].values, buckets=10)
            psi_scores.append((col, psi))
        except Exception:
            continue
    psi_df = pd.DataFrame(psi_scores, columns=["feature", "psi"]).sort_values("psi", ascending=False)
    psi_df.to_csv(os.path.join(psi_dir, "psi_scores.csv"), index=False)

    # Bar chart top N PSI features
    top = psi_df.head(topn)
    fig = plt.figure(figsize=(10,5))
    plt.barh([short_name(c, 30) for c in top["feature"][::-1]], top["psi"][::-1].values)
    plt.xlabel("PSI"); plt.title(f"Top {topn} PSI features (train vs. test)")
    fig.tight_layout()
    fig.savefig(os.path.join(psi_dir, "psi_topn.png"))
    plt.close(fig)

    # Overlay histograms for top 3 drift features
    for feat in top["feature"].head(3):
        fig = plt.figure(figsize=(8,4))
        plt.hist(X_train[feat].dropna().values, bins=50, alpha=0.5, label="train")
        plt.hist(X_test[feat].dropna().values, bins=50, alpha=0.5, label="test")
        plt.title(f"Train vs Test distribution: {feat}")
        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(psi_dir, f"dist_train_vs_test_{safe_name(feat)}.png"))
        plt.close(fig)

    return {
        "psi_csv": "psi/psi_scores.csv",
        "psi_topn_png": "psi/psi_topn.png",
        "psi_overlay_pngs": [f"psi/dist_train_vs_test_{safe_name(f)}.png" for f in top["feature"].head(3)]
    }

# Model defs & tuning

def get_models_and_spaces(random_state=42):
    lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=random_state, solver="liblinear")
    rf = RandomForestClassifier(class_weight="balanced", n_estimators=300, random_state=random_state, n_jobs=-1)
    nb = GaussianNB()

    # Keep it simple:
    lr_grid = {"C": [0.1, 0.5, 1.0, 2.0, 5.0]}
    rf_space = {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 8, 16, 24, 32],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", 0.4, 0.6]
    }
    nb_grid = {"var_smoothing": np.logspace(-12, -6, 7)}

    return (("Logistic Regression", lr, lr_grid),
            ("Random Forest", rf, rf_space),
            ("GaussianNB", nb, nb_grid))

def tune_model(name, model, space, X, y, scoring="roc_auc", random_state=42):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    if name == "Random Forest":
        search = RandomizedSearchCV(model, space, n_iter=20, scoring=scoring, cv=cv, n_jobs=-1, random_state=random_state, verbose=0)
    else:
        search = GridSearchCV(model, space, scoring=scoring, cv=cv, n_jobs=-1, verbose=0)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_

# Evaluation & plots

def evaluate_model(model, X_tr, y_tr, X_te, y_te, label: str, out_dir: str):
    # Predict probabilities (handle classifiers without predict_proba if needed)
    if hasattr(model, "predict_proba"):
        p_tr = model.predict_proba(X_tr)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]
    else:
        # fallback
        p_tr = model.decision_function(X_tr)
        p_te = model.decision_function(X_te)

    yhat_tr = (p_tr >= 0.5).astype(int)
    yhat_te = (p_te >= 0.5).astype(int)

    # Metrics
    metrics = {
        "roc_auc_train": roc_auc_score(y_tr, p_tr),
        "roc_auc_test": roc_auc_score(y_te, p_te),
        "pr_auc_train": average_precision_score(y_tr, p_tr),
        "pr_auc_test": average_precision_score(y_te, p_te),
        "brier_train": brier_score_loss(y_tr, p_tr),
        "brier_test": brier_score_loss(y_te, p_te),
        "f1_train": f1_score(y_tr, yhat_tr, zero_division=0),
        "f1_test": f1_score(y_te, yhat_te, zero_division=0),
        "precision_train": precision_score(y_tr, yhat_tr, zero_division=0),
        "precision_test": precision_score(y_te, yhat_te, zero_division=0),
        "recall_train": recall_score(y_tr, yhat_tr, zero_division=0),
        "recall_test": recall_score(y_te, yhat_te, zero_division=0),
    }

    # Plots (ROC, PR, Calibration) — overlay train vs test
    plots_dir = ensure_dir(os.path.join(out_dir, "plots", safe_name(label)))

    # ROC
    fpr_tr, tpr_tr, _ = roc_curve(y_tr, p_tr)
    fpr_te, tpr_te, _ = roc_curve(y_te, p_te)
    fig = plt.figure(figsize=(6,5))
    plt.plot(fpr_tr, tpr_tr, label=f"Train (AUC={metrics['roc_auc_train']:.3f})")
    plt.plot(fpr_te, tpr_te, label=f"Test (AUC={metrics['roc_auc_test']:.3f})")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {label}"); plt.legend()
    fig.tight_layout()
    roc_path = os.path.join(plots_dir, "roc.png")
    fig.savefig(roc_path); plt.close(fig)

    # Precision-Recall
    pr_tr, rc_tr, _ = precision_recall_curve(y_tr, p_tr)
    pr_te, rc_te, _ = precision_recall_curve(y_te, p_te)
    fig = plt.figure(figsize=(6,5))
    plt.plot(rc_tr, pr_tr, label=f"Train (AP={metrics['pr_auc_train']:.3f})")
    plt.plot(rc_te, pr_te, label=f"Test (AP={metrics['pr_auc_test']:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall — {label}"); plt.legend()
    fig.tight_layout()
    pr_path = os.path.join(plots_dir, "pr.png")
    fig.savefig(pr_path); plt.close(fig)

    # Calibration curve
    frac_pos_tr, mean_pred_tr = calibration_curve(y_tr, p_tr, n_bins=10, strategy="quantile")
    frac_pos_te, mean_pred_te = calibration_curve(y_te, p_te, n_bins=10, strategy="quantile")
    fig = plt.figure(figsize=(6,5))
    plt.plot([0,1],[0,1],"k--")
    plt.plot(mean_pred_tr, frac_pos_tr, marker="o", label=f"Train (Brier={metrics['brier_train']:.3f})")
    plt.plot(mean_pred_te, frac_pos_te, marker="o", label=f"Test (Brier={metrics['brier_test']:.3f})")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    plt.title(f"Calibration — {label}"); plt.legend()
    fig.tight_layout()
    cal_path = os.path.join(plots_dir, "calibration.png")
    fig.savefig(cal_path); plt.close(fig)

    return metrics, {
        "roc": os.path.relpath(roc_path, out_dir),
        "pr": os.path.relpath(pr_path, out_dir),
        "calibration": os.path.relpath(cal_path, out_dir)
    }

# SHAP for interpretability

def shap_summary(best_label, best_model, X_train, out_dir: str, max_samples=2000):
    shap_dir = ensure_dir(os.path.join(out_dir, "shap"))
    try:
        import shap  # optional
        # sample for speed
        if len(X_train) > max_samples:
            X_sample = X_train.sample(n=max_samples, random_state=42)
        else:
            X_sample = X_train

        # Choose explainer
        if hasattr(best_model, "estimators_"):
            explainer = shap.TreeExplainer(best_model)
        else:
            explainer = shap.Explainer(best_model, X_sample)

        shap_values = explainer(X_sample)

        # Beeswarm
        fig = plt.figure(figsize=(8,6))
        shap.plots.beeswarm(shap_values, show=False, max_display=20)
        fig.tight_layout()
        beeswarm_path = os.path.join(shap_dir, f"shap_beeswarm_{safe_name(best_label)}.png")
        plt.savefig(beeswarm_path); plt.close(fig)

        # Bar
        fig = plt.figure(figsize=(8,6))
        shap.plots.bar(shap_values, show=False, max_display=20)
        fig.tight_layout()
        bar_path = os.path.join(shap_dir, f"shap_bar_{safe_name(best_label)}.png")
        plt.savefig(bar_path); plt.close(fig)

        return {"beeswarm": os.path.relpath(beeswarm_path, out_dir),
                "bar": os.path.relpath(bar_path, out_dir)}
    except Exception as e:
        # No shap installed or other issues
        with open(os.path.join(shap_dir, "shap_skipped.txt"), "w") as f:
            f.write(f"SHAP skipped: {e}\nInstall shap to enable (pip install shap).")
        return {"skipped": "shap/shap_skipped.txt"}

# Report (Markdown)

def write_report(out_dir: str, meta: dict, eda_assets: dict, psi_assets: dict, model_summaries: dict,
                 metrics_table: pd.DataFrame, best_block: dict, fs_cols: list, dropped_cols: list):
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Bankruptcy Risk Modeling – Training Pipeline Report\n")
        f.write(f"*Generated:* {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## EDA (Jot Notes)\n")
        f.write("- Numeric financial ratios only → no encoding needed.\n")
        f.write("- Strong class imbalance (~3% bankrupt) → plan class weighting & stratification.\n")
        f.write("- Detected highly correlated ratios → apply correlation filtering (|r|>0.95).\n")
        f.write("- Outliers present; keep unless clear errors, as they may signal distress.\n\n")

        f.write("## Data Preprocessing (Jot Notes)\n")
        f.write("- Median imputation for missing values (robust to outliers).\n")
        f.write("- Standardize features for LR & NB; RF uses raw numeric features.\n")
        f.write("- Imbalance handled with `class_weight='balanced'` for LR & RF.\n")
        f.write("- Stratified train/test split; PSI checks for sampling bias.\n\n")

        f.write("## Feature Selection (Jot Notes)\n")
        f.write("- Simple correlation filter: drop one of any pair with |r|>0.95.\n")
        f.write("- Keep the feature with stronger correlation to target.\n")
        f.write("- Reduces multicollinearity for LR; minimal impact on RF.\n")
        f.write(f"- Selected {len(fs_cols)} features (dropped {len(dropped_cols)}).\n\n")

        f.write("## Hyperparameter Tuning (Jot Notes)\n")
        f.write("- Keep-it-simple: Grid for LR (C), RandomizedSearch for RF, light grid for NB.\n")
        f.write("- CV: Stratified 5-fold; score: ROC-AUC.\n")
        f.write("- Balance compute and performance (n_iter=20 for RF).\n")
        f.write("- Save best params for reproducible training.\n\n")

        f.write("## Model Training (Jot Notes)\n")
        f.write("- Models: Logistic Regression (benchmark), Random Forest, GaussianNB.\n")
        f.write("- Use CV during tuning; refit best estimator on full train.\n")
        f.write("- Save fitted models & scaler to disk for deployment.\n")
        f.write("- Consistent pipeline across models for fair comparison.\n\n")

        f.write("## Model Evaluation & Comparison (Jot Notes)\n")
        f.write("- Compare ROC-AUC, PR-AUC, F1, Precision, Recall, Brier on train & test.\n")
        f.write("- Plot ROC, PR, and Calibration with train/test overlay per model.\n")
        f.write("- Watch AUC gaps & calibration for over/underfitting.\n")
        f.write("- Select best by test ROC-AUC (tie-breaker: PR-AUC & Brier).\n\n")

        f.write("## SHAP Interpretability (Jot Notes)\n")
        f.write("- Compute SHAP for the best test model (TreeExplainer for RF).\n")
        f.write("- Provide beeswarm/bar plots (top drivers of risk).\n")
        f.write("- Align top features with finance intuition (leverage, profitability, liquidity).\n")
        f.write("- Supports regulatory explainability & risk review.\n\n")

        f.write("## PSI / Drift (Jot Notes)\n")
        f.write("- PSI(train vs test) per feature; high PSI suggests sampling bias.\n")
        f.write("- Review top drifted features + overlay distributions.\n")
        f.write("- If PSI > 0.25 repeatedly → re-sample or re-split; monitor in prod.\n")
        f.write("- Adds guardrail for stable deployment.\n\n")

        f.write("## Challenges & Reflections (Jot Notes)\n")
        f.write("- Heavy class imbalance → precision/recall trade-offs by model.\n")
        f.write("- Many correlated ratios → needed simple filtering to stabilize LR.\n")
        f.write("- Calibration varies; Brier score/curves reveal miscalibration.\n")
        f.write("- SHAP on trees can be slow; sub-sampled to keep runtime reasonable.\n\n")

        f.write("## Selected Features\n")
        f.write(f"- Count: {len(fs_cols)}\n")
        f.write(f"- Dropped: {len(dropped_cols)}\n\n")

        f.write("### Metrics Summary (Train/Test)\n\n")
        f.write(metrics_table.to_markdown(index=False))
        f.write("\n\n")

        f.write("### Best Model\n")
        f.write(json.dumps(best_block, indent=2))
        f.write("\n\n")

        f.write("### Key Artifacts\n")
        f.write(f"- EDA describe: `{eda_assets['describe_csv']}`\n")
        f.write(f"- EDA missing: `{eda_assets['missing_csv']}`\n")
        f.write(f"- EDA correlation heatmap: `{eda_assets['corr_heatmap']}`\n")
        f.write(f"- Class balance: `{eda_assets['class_balance']}`\n")
        f.write(f"- PSI scores: `{psi_assets['psi_csv']}`\n")
        f.write(f"- PSI topN: `{psi_assets['psi_topn_png']}`\n")
        f.write("\n")
        for model_name, assets in model_summaries.items():
            f.write(f"- {model_name} ROC: `{assets['roc']}`\n")
            f.write(f"- {model_name} PR: `{assets['pr']}`\n")
            f.write(f"- {model_name} Calibration: `{assets['calibration']}`\n")

    return report_path

# Main pipeline

def main(args):
    out_dir = ensure_dir(args.out_dir)

    # Load data
    df = pd.read_csv(args.data_path)
    if "Bankrupt?" not in df.columns:
        raise ValueError("Target column 'Bankrupt?' not found in CSV.")

    y = df["Bankrupt?"].astype(int)
    X = df.drop(columns=["Bankrupt?"])

    # EDA
    eda_assets = eda_plots(X, y, out_dir)

    # Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # Preprocess: impute medians
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imp  = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Feature selection: correlation filter
    selected_cols, dropped_cols = correlation_filter(X_train_imp, y_train, threshold=0.95)

    # Remove constant columns to avoid divide warnings
    X_train_fs = X_train_imp[selected_cols].copy()

    # drop constant columns
    X_train_fs = X_train_fs.loc[:, X_train_fs.nunique() > 1]  

    X_test_fs  = X_test_imp[selected_cols].copy()

    # keep same cols as train
    X_test_fs  = X_test_fs.loc[:, X_test_fs.columns.isin(X_train_fs.columns)]  

    # Scaler for LR & NB
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fs)
    X_test_scaled  = scaler.transform(X_test_fs)

    # PSI report on selected features
    psi_assets = psi_report(X_train_fs, X_test_fs, out_dir)

    # Models & tuning
    models = get_models_and_spaces(random_state=args.random_state)

    metrics_rows = []
    model_plot_assets = {}
    saved_models = {}
    best_by_auc = {"label": None, "auc": -np.inf, "model_path": None}

    for label, base_model, space in models:
        print(f"\n=== Tuning: {label} ===")
        # Use scaled inputs for LR/NB; raw for RF
        if label in ("Logistic Regression", "GaussianNB"):
            X_tune = X_train_scaled
        else:
            X_tune = X_train_fs.values

        best_est, best_params, best_cv = tune_model(label, base_model, space, X_tune, y_train)
        print(f"Best params: {best_params} | CV ROC-AUC: {best_cv:.4f}")

        # Fit final on train
        if label in ("Logistic Regression", "GaussianNB"):
            best_est.fit(X_train_scaled, y_train)
            metrics, assets = evaluate_model(best_est, X_train_scaled, y_train, X_test_scaled, y_test, label, out_dir)
        else:
            best_est.fit(X_train_fs.values, y_train)
            metrics, assets = evaluate_model(best_est, X_train_fs.values, y_train, X_test_fs.values, y_test, label, out_dir)

        model_plot_assets[label] = assets

        # Save model
        mdl_dir = ensure_dir(os.path.join(out_dir, "models"))
        mdl_path = os.path.join(mdl_dir, f"{safe_name(label)}.joblib")
        joblib.dump(best_est, mdl_path)
        saved_models[label] = {"path": os.path.relpath(mdl_path, out_dir), "best_params": best_params}

        # Track best by test ROC-AUC
        if metrics["roc_auc_test"] > best_by_auc["auc"]:
            best_by_auc = {"label": label, "auc": metrics["roc_auc_test"], "model_path": saved_models[label]["path"]}

        # collect for table
        row = {"model": label, **metrics}
        metrics_rows.append(row)

    # Save scaler & imputer & selected feature list (for deployment)
    prep_dir = ensure_dir(os.path.join(out_dir, "prep"))
    joblib.dump(imputer, os.path.join(prep_dir, "imputer.joblib"))
    joblib.dump(scaler, os.path.join(prep_dir, "scaler.joblib"))
    pd.Series(selected_cols).to_csv(os.path.join(prep_dir, "selected_features.csv"), index=False)

    # Build metrics table
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(out_dir, "metrics_table.csv"), index=False)

    # SHAP on best model
    if best_by_auc["label"] in ("Logistic Regression", "GaussianNB"):
        X_for_shap = pd.DataFrame(X_train_scaled, columns=selected_cols)
    else:
        X_for_shap = X_train_fs

    shap_assets = shap_summary(best_by_auc["label"], joblib.load(os.path.join(out_dir, best_by_auc["model_path"])), X_for_shap, out_dir)

    # Best block for report
    best_block = {
        "best_model": best_by_auc["label"],
        "test_roc_auc": round(best_by_auc["auc"], 4),
        "model_path": best_by_auc["model_path"],
        "prep": {
            "imputer": "prep/imputer.joblib",
            "scaler": "prep/scaler.joblib",
            "selected_features_csv": "prep/selected_features.csv"
        },
        "shap_assets": shap_assets,
        "saved_models": saved_models
    }

    # Write markdown report
    report_path = write_report(out_dir, {}, eda_assets, psi_assets, model_plot_assets, metrics_df, best_block, selected_cols, dropped_cols)
    print(f"\nReport written to: {report_path}")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data.csv")
    parser.add_argument("--out_dir", type=str, default="artifacts")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    main(args)