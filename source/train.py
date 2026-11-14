from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


project_root = Path(__file__).resolve().parents[1]
splits_dir = project_root / "data" / "splits"

train = pd.read_csv(splits_dir / "train.csv")
validation = pd.read_csv(splits_dir / "validation.csv")

target = "closed_before_3y"

X_train = train.drop(columns=target)
y_train = train[target]

X_val = validation[X_train.columns]
y_val = validation[target]

category_columns = ["cuisine", "location_type"]
number_columns = X_train.columns.drop(category_columns).tolist()

# Для логистической регрессии кодируем категории и масштабируем числа.
logistic_preprocessing = ColumnTransformer([
    ("categories", OneHotEncoder(handle_unknown="ignore"), category_columns),
    ("numbers", StandardScaler(), number_columns),
])

# Для случайного леса числовые признаки можно передать как есть.
forest_preprocessing = ColumnTransformer([
    ("categories", OneHotEncoder(handle_unknown="ignore"), category_columns),
    ("numbers", "passthrough", number_columns),
])

models = {
    "Dummy": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": make_pipeline(
        logistic_preprocessing,
        LogisticRegression(max_iter=1000),
    ),
    "Random Forest": make_pipeline(
        forest_preprocessing,
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
    ),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    scores = model.predict_proba(X_val)[:, 1]

    results.append({
        "model": name,
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(y_val, predictions, zero_division=0),
        "recall": recall_score(y_val, predictions, zero_division=0),
        "f1": f1_score(y_val, predictions, zero_division=0),
        "average_precision": average_precision_score(y_val, scores),
    })

results_table = pd.DataFrame(results).sort_values(
    "average_precision", ascending=False
)

print("\nСравнение моделей на validation:")
print(results_table.round(4).to_string(index=False))

# Исходная логистическая регрессия уже обучена в цикле выше.
original = models["Logistic Regression"]

# Делим train на 5 частей и проверяем каждый вариант C.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(
    estimator=original,
    param_grid={
        "logisticregression__C": [0.001, 0.01, 0.1, 1.0, 10.0],
    },
    scoring="average_precision",
    cv=cv,
)

search.fit(X_train, y_train)
tuned = search.best_estimator_

# Сравниваем обе версии на одних и тех же ресторанах из validation.
original_scores = original.predict_proba(X_val)[:, 1]
tuned_scores = tuned.predict_proba(X_val)[:, 1]

original_ap = average_precision_score(y_val, original_scores)
tuned_ap = average_precision_score(y_val, tuned_scores)

print("\n=== Настройка Logistic Regression ===")
print(f"Лучший C по кросс-валидации на train: {search.best_params_['logisticregression__C']}")
print(f"AP исходной модели на validation: {original_ap:.4f}")
print(f"AP настроенной модели на validation: {tuned_ap:.4f}")

if tuned_ap > original_ap:
    final_model = tuned
    final_scores = tuned_scores
    print("Выбрана настроенная модель")
else:
    final_model = original
    final_scores = original_scores
    print("Выбрана исходная модель")

# Проверяем, как порог влияет на precision, recall и F1.
threshold_results = []

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    predicted = (final_scores >= threshold).astype(int)

    threshold_results.append({
        "threshold": threshold,
        "precision": precision_score(y_val, predicted, zero_division=0),
        "recall": recall_score(y_val, predicted, zero_division=0),
        "f1": f1_score(y_val, predicted, zero_division=0),
    })

threshold_table = pd.DataFrame(threshold_results).sort_values(
    "f1", ascending=False, kind="stable"
)
best_threshold = float(threshold_table.iloc[0]["threshold"])

print("\n=== Сравнение порогов на validation ===")
print(threshold_table.round(4).to_string(index=False))
print(f"Выбранный порог: {best_threshold:.2f}")

# Сохраняем модель вместе с обработкой признаков и выбранным порогом.
artifacts_dir = project_root / "artifacts"
artifacts_dir.mkdir(exist_ok=True)

model_path = artifacts_dir / "model.joblib"
joblib.dump(
    {
        "model": final_model,
        "features": X_train.columns.tolist(),
        "threshold": best_threshold,
    },
    model_path,
)
print(f"Модель сохранена: {model_path}")