# Predictive Modeling Using Machine Learning - Breast Tumour Diagnosis

**Goal:** build and evaluate supervised-learning models that predict whether a breast tumour is **malignant or benign** from measurements of cell nuclei.
**Tools:** Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn
**Algorithms:** Logistic Regression, Decision Tree, Random Forest (default and tuned), Gradient Boosting

> This is a learning project. It is **not** a medical diagnostic tool.

---

## 1. Dataset
Wisconsin Diagnostic Breast Cancer dataset (UCI repository, bundled with scikit-learn): **569 patients, 30 numeric features** (radius, texture, perimeter, area, smoothness, concavity, etc., each as mean, error and "worst" value).
- Target: **diagnosis** (1 = malignant, 0 = benign). I flipped the original coding so that malignant is the positive class, which makes recall mean "how many cancers did we catch".
- 357 benign (62.7%) and 212 malignant (37.3%): moderately imbalanced, so I used **stratified** splits.
- Data quality check: 0 missing values and 0 duplicates.

![EDA](outputs/01_eda_class_and_correlation.png)
![Distributions](outputs/02_eda_feature_distributions.png)

## 2. Method
1. **Split:** 80% train (455) / 20% test (114), stratified, fixed random seed (42).
2. **Models:** Logistic Regression (with feature scaling), Decision Tree, Random Forest, Gradient Boosting.
3. **Validation:** 5-fold stratified cross-validation on the training set only.
4. **Tuning:** `GridSearchCV` on Random Forest (36 parameter combinations, optimising recall).
5. **Evaluation on the untouched test set:** accuracy, precision, recall, F1, ROC-AUC, confusion matrices and ROC curves.
6. **Final model chosen by cross-validation score, not test score**, so the test set stays an honest, unseen check.
7. **Decision threshold** chosen on cross-validated training predictions, then tested once on the test set.

Run it yourself: `pip install -r requirements.txt` then `python predictive_modeling.py`.

## 3. Results

| Model | CV Accuracy | Train Acc | Test Acc | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **97.4%** | 98.7% | 96.5% | 97.5% | 92.9% | 95.1% | **0.996** |
| Decision Tree | 92.3% | 100% | 93.0% | 90.5% | 90.5% | 90.5% | 0.925 |
| Random Forest | 96.3% | 100% | 96.5% | 100% | 90.5% | 95.0% | 0.994 |
| Gradient Boosting | 96.7% | 100% | 96.5% | 100% | 90.5% | 95.0% | 0.995 |
| Random Forest (tuned) | 96.5% | 100% | 95.6% | 100% | 88.1% | 93.7% | 0.995 |

![Comparison](outputs/03_model_comparison.png)
![Confusion matrices](outputs/04_confusion_matrices.png)
![ROC curves](outputs/05_roc_curves.png)
![Train vs test](outputs/07_train_vs_test.png)

**Final model: Logistic Regression**, which had the best cross-validated accuracy (97.4%) and the highest test ROC-AUC (0.996).

### Improving recall with the decision threshold
At the default threshold of 0.50 the final model missed 3 of 42 malignant tumours in the test set. A missed cancer (false negative) is much worse than a false alarm, so I lowered the threshold using **only training cross-validation** (rule: highest threshold that keeps recall at or above 97%). That gave **0.31**.

| Threshold | Accuracy | Precision | Recall | F1 | Missed cancers (FN) | False alarms (FP) |
|---|---|---|---|---|---|---|
| 0.50 (default) | 96.5% | 97.5% | 92.9% | 95.1% | 3 | 1 |
| **0.31 (tuned)** | **98.2%** | 97.6% | **97.6%** | **97.6%** | **1** | 1 |

![Threshold trade-off](outputs/09_threshold_tradeoff.png)
![Final confusion](outputs/10_final_confusion_threshold.png)

## 4. Key insights
1. **A simple model won.** Logistic Regression matched or beat the ensemble models. The classes are almost linearly separable, so extra complexity brought no benefit.
2. **The decision tree overfits.** It scores 100% on training data but only 93% on test data, the largest gap of any model. Random Forest fixes much of that by averaging many trees.
3. **Hyperparameter tuning did not help here.** The tuned Random Forest was not better than the default one on the test set. The difference is a few samples and within noise, which is a useful lesson: tuning is not guaranteed to improve results.
4. **Accuracy alone is misleading.** The Random Forest has 100% precision but misses about 1 in 10 cancers. Recall matters more in this problem, and moving the threshold improved it more than model tuning did.
5. **The most important signals** are the "worst" (largest) tumour measurements: *worst concave points, worst area, worst radius, mean perimeter, mean concave points* (Random Forest importance, chart below). Larger and more irregular nuclei indicate malignancy.

![Feature importance](outputs/06_feature_importance.png)
![Logistic coefficients](outputs/11_logistic_coefficients.png)
![Decision tree](outputs/08_decision_tree_depth3.png)

## 5. Limitations
- The test set has only 114 patients, so one patient changes accuracy by about 0.9%. The model ranking is indicative, not definitive. A different random split can change the order of the top models.
- The threshold is tuned for recall at the cost of a few more false alarms, which is a business/clinical choice.
- The dataset is small and comes from one source; the model has not been validated on new hospitals or populations.
- Not for real medical use.

## 6. Files
```
data/breast_cancer_data.csv       dataset (569 x 31)
predictive_modeling.py            full pipeline
requirements.txt                  dependencies
outputs/                          charts, model_comparison.csv, threshold_comparison.csv,
                                  classification_report.txt, summary.json, sample_predictions.csv,
                                  final_model.joblib (saved model + threshold)
```
**Using the saved model:**
```python
import joblib, pandas as pd
bundle = joblib.load("outputs/final_model.joblib")
X = pd.read_csv("data/breast_cancer_data.csv").drop(columns="diagnosis")[bundle["features"]]
prob = bundle["model"].predict_proba(X.head())[:, 1]
print(prob >= bundle["threshold"])      # True = predicted malignant
```

## 7. Skills demonstrated
Supervised learning (classification), train/test splitting, stratification, cross-validation, hyperparameter tuning (GridSearchCV), overfitting diagnosis, evaluation with confusion matrix / ROC-AUC / precision / recall / F1, decision-threshold selection, feature importance and model interpretation, model persistence, and communicating results.
