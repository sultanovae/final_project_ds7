from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


project_root = Path(__file__).resolve().parents[1]
splits_dir = project_root / "data" / "splits_v2"

train = pd.read_csv(splits_dir / "train.csv")
validation = pd.read_csv(splits_dir / "validation.csv")

target = "closed_before_3y"
features = [
    "location_type",
    "is_franchise",
    "owner_experience_years",
    "starting_capital_months",
    "foot_traffic",
    "competition_density",
    "has_marketing",
]

categories = ["location_type"]
numbers = [column for column in features if column not in categories]

preprocessing = ColumnTransformer([
    ("categories", OneHotEncoder(handle_unknown="ignore"), categories),
    ("numbers", StandardScaler(), numbers),
])

model = make_pipeline(
    preprocessing,
    LogisticRegression(C=0.001, max_iter=1000),
)

model.fit(train[features], train[target])

scores = pd.Series(
    model.predict_proba(validation[features])[:, 1],
    index=validation.index,
)

# Границы рассчитываем по оценкам модели на validation.
low_cutoff = float(scores.quantile(0.30))
high_cutoff = float(scores.quantile(0.80))

# Равенство границе относит заведение к следующему уровню.
levels = pd.cut(
    scores,
    bins=[float("-inf"), low_cutoff, high_cutoff, float("inf")],
    labels=["Низкий", "Средний", "Высокий"],
    right=False,
)

summary = (
    validation.assign(risk_level=levels)
    .groupby("risk_level", observed=False)[target]
    .agg(restaurants="size", closed="sum", closed_share="mean")
)

summary["group_share_pct"] = summary["restaurants"] / len(validation) * 100
summary["closed_share"] *= 100
summary["share_of_all_closures_pct"] = (
    summary["closed"] / validation[target].sum() * 100
)

ap = average_precision_score(validation[target], scores)

print(f"AP короткой формы на validation: {ap:.4f}")
print(f"Граница низкого и среднего уровня: {low_cutoff:.6f}")
print(f"Граница среднего и высокого уровня: {high_cutoff:.6f}")
print("\n=== Уровни риска на validation ===")
print(summary.round(2).to_string())

artifacts_dir = project_root / "artifacts"
artifacts_dir.mkdir(exist_ok=True)

model_path = artifacts_dir / "model_v2.joblib"
joblib.dump(
    {
        "model": model,
        "features": features,
        "risk_cutoffs": [low_cutoff, high_cutoff],
        "capital_range": [
    float(train["starting_capital_months"].min()),
    float(train["starting_capital_months"].max()),
],
    },
    model_path,
)
print(f"\nМодель и границы сохранены: {model_path}")
