"""
Employee Attrition ML Prediction
--------------------------------
Adds supervised ML to the existing Power BI employee attrition analysis.

Models:
1. Logistic Regression - interpretable baseline and selected risk-scoring model
2. Random Forest - nonlinear benchmark

Outputs:
- ml_outputs/model_comparison.csv
- ml_outputs/employee_attrition_predictions.csv
- ml_outputs/logistic_feature_importance.csv
- ml_outputs/model_metrics.txt
"""

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "Employee_Attrition_Dataset.csv"
OUT = BASE / "ml_outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Target: 1 = employee likely to leave, 0 = stays
y = (df["Attrition"] == "Yes").astype(int)

# Remove target, employee identifier and constant columns.
# EmployeeNumber is an identifier, not a predictive business feature.
drop_cols = ["Attrition", "EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
X = df.drop(columns=drop_cols)

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_cols,
        ),
    ]
)

models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )),
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )),
    ]),
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

metrics = []
fitted_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, prob),
    })

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(OUT / "model_comparison.csv", index=False)

# Logistic Regression is selected for employee risk scoring because
# recall and interpretability are more useful than raw accuracy for HR intervention.
risk_model = fitted_models["Logistic Regression"]

# Score every employee so Power BI can consume the ML output.
all_prob = risk_model.predict_proba(X)[:, 1]
all_pred = (all_prob >= 0.50).astype(int)

predictions = df.copy()
predictions["Attrition_Probability"] = all_prob.round(4)
predictions["ML_Attrition_Prediction"] = np.where(all_pred == 1, "Yes", "No")
predictions["Risk_Band"] = pd.cut(
    all_prob,
    bins=[-0.01, 0.30, 0.60, 1.00],
    labels=["Low", "Medium", "High"]
)

# Keep the original business fields plus ML outputs.
predictions.to_csv(OUT / "employee_attrition_predictions.csv", index=False)

# Extract interpretable logistic-regression coefficients.
prep = risk_model.named_steps["preprocessor"]
model = risk_model.named_steps["model"]
feature_names = prep.get_feature_names_out()

importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": model.coef_[0],
})
importance["Absolute_Impact"] = importance["Coefficient"].abs()
importance["Direction"] = np.where(
    importance["Coefficient"] > 0,
    "Increases attrition risk",
    "Decreases attrition risk"
)
importance = importance.sort_values("Absolute_Impact", ascending=False)
importance.to_csv(OUT / "logistic_feature_importance.csv", index=False)

# Save readable evaluation summary.
with open(OUT / "model_metrics.txt", "w", encoding="utf-8") as f:
    f.write("EMPLOYEE ATTRITION ML MODEL EVALUATION\n")
    f.write("=" * 45 + "\n\n")
    f.write(metrics_df.to_string(index=False))
    f.write("\n\nSelected model: Logistic Regression\n")
    f.write("Reason: stronger recall and interpretable risk drivers for HR intervention.\n\n")
    f.write("Confusion Matrix - Logistic Regression\n")
    f.write(str(confusion_matrix(y_test, risk_model.predict(X_test))))
    f.write("\n\nClassification Report - Logistic Regression\n")
    f.write(classification_report(
        y_test, risk_model.predict(X_test), target_names=["Stay", "Leave"], zero_division=0
    ))

print("ML pipeline completed.")
print(metrics_df.to_string(index=False))
print(f"\nOutputs saved to: {OUT}")
