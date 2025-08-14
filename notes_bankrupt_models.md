"""
Lab 04 – Binary Classification: Company Bankruptcy Prediction  
Author: Dylan Joy  
Notes: This file contains the structured breakdown of key decision-making areas.  
A separate file will hold graphs and visualizations.  
"""


# 1. Choosing the Initial Models

- Logistic Regression: baseline, interpretable  
- Random Forest: non-linear, robust to outliers, works well on tabular data  
- GaussianNB (Naive Bayes): fast probabilistic model w/ performance contrast  

- Used all 3 models simulataneously for comparison within visualizations  
- Initially used Logistic regression for simplicity and stragithforward.  
- Random Forest and GaussianNB helped capture the non-linear relationships between features  
- Avoided clustering: not suitable for labeled classification  
- Unsupervised clustering would be best if labelling is incomplete/noisy  
- Supervised is better for these models, as we do have effectively labeled data  


# 2. Data Pre-processing

- Logistic Regression and Naive Bayes are senstitive to feature scales  
- StandardScaler ensured feature has a mean of 0 and a standard deviation of 1  
- Skipped scaling for RF, not distant based  
- Data already numeric, no categorical encoding required  
- Reduced computation time by preprocessing each model's matched requirements  


# 3. Handling Class Imbalance

- Imbalance ~3% bankrupt  
- Used class_weight='balanced' for LR & RF  
- Stratified train-test split for fair representation  
- Stratified train-test split (80/20): preserves class imbalance ratios  
- Important so minority bankruptcy cases are proportionally represented  

- Avoided SMOTE to prevent overfitting on small dataset  
- SMOTE: balances data w/ synthetic samples (recall, risk overfit/distorted space)  
- Undersampling: trims majority (simple, risk losing info/underfit)  
- Class weighting: adjusts loss for imbalance (keeps data, risk overcompensating)  

- Chose stratification + class_weight='balanced' (LR, RF)  
- Avoided SMOTE/undersampling: data size sufficient, wanted to avoid synthetic distortion  


# 4. Outlier Detection and Treatment

- Outliers can distort model training and affect minority class detection  
- Outliers may indicate financial distress — important signals  
- No aggressive outlier removal, could blindly lose important information  


# 5. Addressing Sampling Bias

- Stratified split maintains minority/majority proportions  
- Important to ensure train and test sets avoid biased performance  
- No PSI check in lab — but useful for detecting feature drift  


# 6. Data Normalization

- StandardScaler applied to Logistic Regression & Naive Bayes b/c of normalization  
- Random Forest left unscaled  
- mean = 0 and std = 1  
- Normalization was applied selectively per model for effficiency/performance  


# 7. Testing for Normality

- Not required for Random Forest  
- Logistic Regression & Naive Bayes benefit from normality  
- Skewed financial ratios are common for these, log or Box-Cox transforms could be useful  
- Scaling mitigates issues, No explicit normality/transformation done  


# 8. Dimensionality Reduction (PCA)

- PCA can reduce overfitting and has a risk of losing interpretability of financial ratios  
- PCA not used — dataset not high-dimensional  
- Would be considered if more features were present  


# 9. Feature Engineering

- Dataset has many financial ratios, considered sufficient for model foundation  
- No new features added to focus on model comparison — dataset already domain-specific  
- Future work / advanced modelling like temporal trends or economic indicators are available  
- Can easily be updated for next steps  


# 10. Testing and Addressing Multicollinearity

- No explicit removal — Random Forest tolerates correlated features  
- Logistic Regression may be affected but used as baseline  
- Model performance suggests no major impact from no explicit multicollinearity removal  
- VIF could impact interpretability & inflate variances  
- Correlation matrix detected through Multicollinearity  


# 11. Feature Selection Methods

- Kept all features due to small dataset, even though too many features risk overfitting  
- Random Forest feature importance could guide future selection  
- No automated feature selection was performed  
- Pruning in the future could be utilized  
- Correlation filtering, model feature importance (RF gain), and regularization (lasso/ridge)  


# 12. Hyperparameter Tuning

- Defaults + class_weight='balanced'  
- Could improve with GridSearchCV or RandomizedSearchCV  
- Chose grid/random search for simplicity  
- Tuning mainly applied to Logistic Regression and Random Forest  
- Balanced computational resources w/ model gains  


# 13. Cross-Validation Strategy

- Imbalanced binary classification is best for maintaining class proportions w/ stratified k-fold cross-validation  
- Single stratified train-test split used  
- Stratified k-fold recommended for robustness  
- Simple holdout used here due to project scope  


# 14. Evaluation Metrics Selection

- Used precision, recall, F1-score over accuracy  
- Plotted ROC and Precision-Recall curves  
- Good overall separability  
- Used predict_proba for curve plotting  
- F1 balances precision/recall but won't hide any imbalances  
- Classification_report on predict(x) for class-wise insight  


# 15. Evaluating Drift and Model Degradation

- PSI not used in lab but important for long-term monitoring  
- Data drift monitoring is essential to detect changes over time  
- Flag shifts by comparing train/test and/or historical/new data via PSI  
- Best practice to split datasets and calculate PSI between the train/test sets  


# 16. Interpreting Model Results

- RF: feature importance for drivers of bankruptcy  
- LR: coefficients show direction of impact  
- ROC & PR curves are stakeholder-friendly  
- SHAP, LIME, and feature importance methods were included  
- SHAP details local and global explanations  
- LIME interprets estimatations for individual predictions  
- RF feature importance provides quick insights w/ less granularity  


# 17. Final Notes

- This file serves as organized documentation of decisions.  
- Graphs and visualizations will be implemented in a separate .py file.  
