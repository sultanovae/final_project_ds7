from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


project_root = Path(__file__).resolve().parents[1]

saved = joblib.load(project_root / "artifacts" / "model.joblib")
model = saved["model"]
features = saved["features"]
threshold = saved["threshold"]

test = pd.read_csv(project_root / "data" / "splits" / "test.csv")
X_test = test[features]
y_test = test["closed_before_3y"]

scores = model.predict_proba(X_test)[:, 1]
predictions = (scores >= threshold).astype(int)

metrics = {
    "accuracy": accuracy_score(y_test, predictions),
    "precision": precision_score(y_test, predictions, zero_division=0),
    "recall": recall_score(y_test, predictions, zero_division=0),
    "f1": f1_score(y_test, predictions, zero_division=0),
    "average_precision": average_precision_score(y_test, scores),
}

matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
tn, fp, fn, tp = matrix.ravel()

print(f"\n=== Итоговая оценка на test · {len(test):,} строк ===")
print(f"Порог: {threshold:.2f}")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nРазбор предсказаний:")
print(f"TN — верно определила, что ресторан достиг трёх лет: {tn}")
print(f"FP — ложная тревога: {fp}")
print(f"FN — пропущенное закрытие: {fn}")
print(f"TP — верно выявила закрытие до трёх лет: {tp}")