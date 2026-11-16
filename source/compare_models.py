from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


project_root = Path(__file__).resolve().parents[1]
splits_dir = project_root / "data" / "splits_v2"

train = pd.read_csv(splits_dir / "train.csv")
validation = pd.read_csv(splits_dir / "validation.csv")

target = "closed_before_3y"
features = [
    "cuisine",
    "location_type",
    "is_franchise",
    "owner_experience_years",
    "starting_capital_months",
    "price_tier",
    "seats",
    "menu_size",
    "foot_traffic",
    "competition_density",
    "has_marketing",
]

X_train, y_train = train[features], train[target]
X_val, y_val = validation[features], validation[target]

categories = ["cuisine", "location_type"]
numbers = [column for column in features if column not in categories]


def make_preprocessing(numeric_transform):
    return ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), categories),
        ("numbers", numeric_transform, numbers),
    ])


logistic = make_pipeline(
    make_preprocessing(StandardScaler()),
    LogisticRegression(C=0.001, max_iter=1000),
)

forest = make_pipeline(
    make_preprocessing("passthrough"),
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
)

search = GridSearchCV(
    estimator=forest,
    param_grid={
        "randomforestclassifier__max_depth": [5, 10, None],
        "randomforestclassifier__min_samples_leaf": [1, 10, 30],
    },
    scoring="average_precision",
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    n_jobs=1,
    verbose=1,
)

top_count = len(validation) // 5
results = []


def add_result(name, model):
    scores = model.predict_proba(X_val)[:, 1]
    ranked = validation[[target]].assign(score=scores).sort_values(
        "score", ascending=False, kind="stable"
    )
    top_group = ranked.head(top_count)

    results.append({
        "Модель": name,
        "AP": average_precision_score(y_val, scores),
        "Закрылись в топ-20%, %": top_group[target].mean() * 100,
        "Закрытий в топ-20%, шт.": int(top_group[target].sum()),
    })


logistic.fit(X_train, y_train)
add_result("Logistic Regression", logistic)

forest.fit(X_train, y_train)
add_result("Random Forest до настройки", forest)

search.fit(X_train, y_train)
add_result("Random Forest после настройки", search.best_estimator_)

print("\nЛучшие параметры леса на train:")
print(search.best_params_)
print(f"Средний AP лучшего леса на фолдах train: {search.best_score_:.4f}")

print("\n=== Сравнение моделей на validation · набор B ===")
print(pd.DataFrame(results).round(4).to_string(index=False))