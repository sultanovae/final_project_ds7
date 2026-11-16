from pathlib import Path

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

features_a = [
    "cuisine",
    "location_type",
    "is_franchise",
    "owner_experience_years",
    "starting_capital_months",
    "price_tier",
    "seats",
    "menu_size",
]

features_b = [
    *features_a,
    "foot_traffic",
    "competition_density",
    "has_marketing",
]

features_c = [
    *features_b,
    "rent_to_revenue",
    "labor_cost_ratio",
]

# Каждый вариант B без одного поля сравниваем с полным B.
feature_sets = {
    "A — 8 исходных": features_a,
    "B — 11 признаков": features_b,
    "C — 13 признаков": features_c,
}

for removed in ["cuisine", "price_tier", "seats", "menu_size", "has_marketing"]:
    feature_sets[f"B без {removed}"] = [
        feature for feature in features_b if feature != removed
    ]

feature_sets["B короткая форма — 7 полей"] = [
    column
    for column in features_b
    if column not in {"cuisine", "price_tier", "seats", "menu_size"}
]

y_train = train[target]
y_val = validation[target]
top_count = len(validation) // 5
results = []

for name, features in feature_sets.items():
    categories = [
        column for column in ["cuisine", "location_type"] if column in features
    ]
    numbers = [column for column in features if column not in categories]

    preprocessing = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), categories),
        ("numbers", StandardScaler(), numbers),
    ])

    model = make_pipeline(
        preprocessing,
        LogisticRegression(C=0.001, max_iter=1000),
    )

    model.fit(train[features], y_train)
    scores = model.predict_proba(validation[features])[:, 1]

    ranked = validation[[target]].assign(score=scores).sort_values(
        "score", ascending=False, kind="stable"
    )
    top_group = ranked.head(top_count)

    results.append({
        "Вариант": name,
        "Признаков": len(features),
        "AP": average_precision_score(y_val, scores),
        "Закрылись в топ-20%, %": top_group[target].mean() * 100,
        "Закрытий в топ-20%, шт.": int(top_group[target].sum()),
    })

print(f"Общая доля закрытий на validation: {y_val.mean():.2%}")
print(f"Заведений в топ-20% для каждого варианта: {top_count:,}")
print(pd.DataFrame(results).round(4).to_string(index=False))