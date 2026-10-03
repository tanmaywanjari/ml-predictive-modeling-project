"""
Predictive Modeling Using Machine Learning
Task: predict whether a breast tumour is MALIGNANT or BENIGN from cell-nucleus measurements.
Dataset: Wisconsin Diagnostic Breast Cancer (569 samples, 30 numeric features), UCI / scikit-learn.

Pipeline: load -> explore -> split -> train 4 models -> cross-validate -> test
          -> tune best model -> evaluate (confusion matrix, ROC) -> feature importance -> save model
Run:  python predictive_modeling.py
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

SEED = 42
OUT = Path("output"); OUT.mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")
COLORS = {"Logistic Regression": "#264653", "Decision Tree": "#e76f51",
          "Random Forest": "#2a9d8f", "Gradient Boosting": "#e9c46a"}

# ============================================================ 1. LOAD
bc = load_breast_cancer(as_frame=True)
df = bc.frame.copy()
# In the original data 0 = malignant, 1 = benign. Flip so that 1 = MALIGNANT (the case we must not miss).
df["diagnosis"] = 1 - df["target"]
df = df.drop(columns="target")
df.to_csv("data/breast_cancer_data.csv", index=False)
print("Shape:", df.shape, "| missing:", int(df.isna().sum().sum()), "| duplicates:", int(df.duplicated().sum()))

X, y = df.drop(columns="diagnosis"), df["diagnosis"]
class_counts = y.map({0: "Benign", 1: "Malignant"}).value_counts()
print(class_counts)

# ============================================================ 2. EXPLORATORY DATA ANALYSIS
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].bar(class_counts.index, class_counts.values, color=["#2a9d8f", "#e76f51"])
for i, v in enumerate(class_counts.values):
    ax[0].text(i, v + 5, f"{v} ({v / len(df):.0%})", ha="center", fontweight="bold")
ax[0].set_title("Class distribution"); ax[0].set_ylabel("Samples")
top_feats = X.corrwith(y).abs().sort_values(ascending=False).head(10).index
sns.heatmap(df[list(top_feats) + ["diagnosis"]].corr(), cmap="coolwarm", center=0, ax=ax[1],
            annot=True, fmt=".2f", annot_kws={"size": 7}, cbar=False)
ax[1].set_title("Correlation - 10 features most linked to diagnosis")
fig.tight_layout(); fig.savefig(OUT / "01_eda_class_and_correlation.png", dpi=150); plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for a, f in zip(axes.ravel(), top_feats[:6]):
    sns.kdeplot(data=df, x=f, hue=df["diagnosis"].map({0: "Benign", 1: "Malignant"}),
                fill=True, common_norm=False, palette=["#2a9d8f", "#e76f51"], ax=a)
    a.set_title(f)
fig.suptitle("Top 6 features - malignant vs benign distributions", fontsize=14, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "02_eda_feature_distributions.png", dpi=150); plt.close(fig)

# ============================================================ 3. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)      # stratify keeps the class ratio equal
print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

# ============================================================ 4. MODELS
models = {
    "Logistic Regression": Pipeline([("scale", StandardScaler()),
                                     ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows, fitted, probs, preds = [], {}, {}, {}
for name, model in models.items():
    cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    model.fit(X_train, y_train)
    p, pr = model.predict(X_test), model.predict_proba(X_test)[:, 1]
    fitted[name], probs[name], preds[name] = model, pr, p
    rows.append({"Model": name,
                 "CV Accuracy (mean)": cv_acc.mean(), "CV Std": cv_acc.std(),
                 "Train Accuracy": accuracy_score(y_train, model.predict(X_train)),
                 "Test Accuracy": accuracy_score(y_test, p),
                 "Precision": precision_score(y_test, p), "Recall": recall_score(y_test, p),
                 "F1": f1_score(y_test, p), "ROC-AUC": roc_auc_score(y_test, pr)})
results = pd.DataFrame(rows).set_index("Model").round(4)
print("\n", results)

# ============================================================ 5. HYPERPARAMETER TUNING (Random Forest)
grid = {"n_estimators": [100, 300], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]}
gs = GridSearchCV(RandomForestClassifier(random_state=SEED), grid, cv=cv, scoring="recall", n_jobs=-1)
gs.fit(X_train, y_train)          # optimise RECALL: missing a malignant tumour is the costly error
best = gs.best_estimator_
bp, bpr = best.predict(X_test), best.predict_proba(X_test)[:, 1]
tuned = {"Model": "Random Forest (tuned)", "CV Accuracy (mean)": cross_val_score(best, X_train, y_train, cv=cv).mean(),
         "CV Std": cross_val_score(best, X_train, y_train, cv=cv).std(),
         "Train Accuracy": accuracy_score(y_train, best.predict(X_train)),
         "Test Accuracy": accuracy_score(y_test, bp), "Precision": precision_score(y_test, bp),
         "Recall": recall_score(y_test, bp), "F1": f1_score(y_test, bp), "ROC-AUC": roc_auc_score(y_test, bpr)}
results.loc["Random Forest (tuned)"] = pd.Series(tuned).drop("Model").round(4)
fitted["Random Forest (tuned)"], probs["Random Forest (tuned)"], preds["Random Forest (tuned)"] = best, bpr, bp
COLORS["Random Forest (tuned)"] = "#8338ec"
print("\nBest params:", gs.best_params_)
results.to_csv(OUT / "model_comparison.csv")

# ============================================================ 6. VISUALISE PERFORMANCE
# 6a model comparison
metrics = ["Test Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
fig, ax = plt.subplots(figsize=(12, 5))
results[metrics].plot.bar(ax=ax, width=0.8, colormap="viridis", rot=15)
ax.set_ylim(0.85, 1.01); ax.set_title("Model comparison on the unseen test set", fontweight="bold")
ax.legend(loc="lower right", ncol=5, fontsize=9); ax.set_ylabel("Score")
fig.tight_layout(); fig.savefig(OUT / "03_model_comparison.png", dpi=150); plt.close(fig)

# 6b confusion matrices
names = list(fitted)
fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.2))
for a, n in zip(axes, names):
    ConfusionMatrixDisplay(confusion_matrix(y_test, preds[n]), display_labels=["Benign", "Malignant"]
                           ).plot(ax=a, cmap="Blues", colorbar=False, values_format="d")
    a.set_title(n, fontsize=11); a.grid(False)
fig.suptitle("Confusion matrices (test set)", fontsize=14, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "04_confusion_matrices.png", dpi=150); plt.close(fig)

# 6c ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for n in names:
    fpr, tpr, _ = roc_curve(y_test, probs[n])
    ax.plot(fpr, tpr, lw=2.2, color=COLORS[n], label=f"{n} (AUC = {roc_auc_score(y_test, probs[n]):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
ax.set_title("ROC curves", fontweight="bold"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "05_roc_curves.png", dpi=150); plt.close(fig)

# 6d feature importance (tuned RF)
imp = pd.Series(best.feature_importances_, index=X.columns).sort_values().tail(12)
fig, ax = plt.subplots(figsize=(8, 5.5))
imp.plot.barh(ax=ax, color="#2a9d8f"); ax.set_title("Top 12 features - tuned Random Forest", fontweight="bold")
ax.set_xlabel("Importance")
fig.tight_layout(); fig.savefig(OUT / "06_feature_importance.png", dpi=150); plt.close(fig)

# 6e overfitting check: train vs test accuracy
fig, ax = plt.subplots(figsize=(9, 4.5))
results[["Train Accuracy", "Test Accuracy"]].plot.bar(ax=ax, color=["#e9c46a", "#264653"], rot=15)
ax.set_ylim(0.85, 1.02); ax.set_title("Train vs test accuracy (overfitting check)", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "07_train_vs_test.png", dpi=150); plt.close(fig)

# 6f small readable decision tree
small = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(X_train, y_train)
fig, ax = plt.subplots(figsize=(16, 7))
plot_tree(small, feature_names=X.columns, class_names=["Benign", "Malignant"], filled=True,
          rounded=True, fontsize=9, ax=ax)
ax.set_title(f"Decision tree (depth 3) - test accuracy {small.score(X_test, y_test):.1%}", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "08_decision_tree_depth3.png", dpi=150); plt.close(fig)

# ============================================================ 7. FINAL MODEL SELECTION
# Choose using CROSS-VALIDATION on the training data only (never the test set, to avoid leakage).
final = results["CV Accuracy (mean)"].idxmax()
fm = fitted[final]
print("\nFinal model (best CV accuracy):", final)

# ---- decision threshold: a missed malignant tumour (false negative) is worse than a false alarm.
# Pick the threshold on TRAIN cross-validated probabilities, then test it once on the test set.
from sklearn.model_selection import cross_val_predict
cv_prob = cross_val_predict(fm, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
ths = np.round(np.arange(0.05, 0.96, 0.01), 2)
rec = np.array([recall_score(y_train, cv_prob >= t) for t in ths])
prec = np.array([precision_score(y_train, cv_prob >= t, zero_division=0) for t in ths])
TARGET_RECALL = 0.97
ok = np.where(rec >= TARGET_RECALL)[0]
thr = float(ths[ok[-1]]) if len(ok) else 0.5       # highest threshold that still keeps recall >= 97%
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(ths, rec, lw=2.2, color="#e76f51", label="Recall"); ax.plot(ths, prec, lw=2.2, color="#264653", label="Precision")
ax.axvline(thr, ls="--", color="grey", label=f"chosen threshold = {thr:.2f}")
ax.axvline(0.5, ls=":", color="black", label="default threshold = 0.50")
ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score (train, cross-validated)")
ax.set_title("Precision-recall trade-off vs decision threshold", fontweight="bold"); ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "09_threshold_tradeoff.png", dpi=150); plt.close(fig)

p_def = (probs[final] >= 0.5).astype(int); p_thr = (probs[final] >= thr).astype(int)
def rep_row(p): return {"accuracy": accuracy_score(y_test, p), "precision": precision_score(y_test, p),
                        "recall": recall_score(y_test, p), "f1": f1_score(y_test, p)}
thr_cmp = pd.DataFrame({"Default threshold 0.50": rep_row(p_def), f"Tuned threshold {thr:.2f}": rep_row(p_thr)}).T.round(4)
print("\n", thr_cmp)
thr_cmp.to_csv(OUT / "threshold_comparison.csv")

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for a, p, t in zip(ax, [p_def, p_thr], ["Threshold 0.50 (default)", f"Threshold {thr:.2f} (tuned)"]):
    ConfusionMatrixDisplay(confusion_matrix(y_test, p), display_labels=["Benign", "Malignant"]).plot(
        ax=a, cmap="Blues", colorbar=False, values_format="d"); a.set_title(t); a.grid(False)
fig.suptitle(f"{final}: effect of the decision threshold (test set)", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "10_final_confusion_threshold.png", dpi=150); plt.close(fig)

# ---- interpret the final model (logistic regression coefficients on standardised features)
if final == "Logistic Regression":
    coef = pd.Series(fm.named_steps["clf"].coef_[0], index=X.columns).sort_values()
    sel = pd.concat([coef.head(6), coef.tail(6)])
    fig, ax = plt.subplots(figsize=(8, 6))
    sel.plot.barh(ax=ax, color=["#2a9d8f" if v < 0 else "#e76f51" for v in sel])
    ax.set_title("Logistic regression coefficients (standardised features)\nred -> pushes toward MALIGNANT, green -> BENIGN", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "11_logistic_coefficients.png", dpi=150); plt.close(fig)

rep = classification_report(y_test, p_thr, target_names=["Benign", "Malignant"])
print("\n", rep)
(OUT / "classification_report.txt").write_text(f"Model: {final}\nThreshold: {thr}\n\n{rep}")
tn, fp, fn, tp = confusion_matrix(y_test, p_thr).ravel()
tn0, fp0, fn0, tp0 = confusion_matrix(y_test, p_def).ravel()
summary = {"final_model": final, "chosen_threshold": thr, "rf_best_params": gs.best_params_,
           "test_confusion_default": {"TN": int(tn0), "FP": int(fp0), "FN": int(fn0), "TP": int(tp0)},
           "test_confusion_tuned_threshold": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
           "rf_top_features": imp.sort_values(ascending=False).head(5).round(4).to_dict(),
           "results": results.to_dict(orient="index")}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
joblib.dump({"model": fm, "threshold": thr, "features": list(X.columns)}, OUT / "final_model.joblib")

# demo: predict on a few unseen rows with the saved model
sample = X_test.head(8); pr = fm.predict_proba(sample)[:, 1]
demo = pd.DataFrame({"Predicted": np.where(pr >= thr, "Malignant", "Benign"),
                     "Malignant probability": pr.round(3),
                     "Actual": np.where(y_test.head(8) == 1, "Malignant", "Benign")})
print("\nSample predictions:\n", demo)
demo.to_csv(OUT / "sample_predictions.csv", index=False)
print("\nDone -> output/")
