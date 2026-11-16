from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parents[1]
train = pd.read_csv(project_root / "data" / "splits_v2" / "train.csv")
target = "closed_before_3y"

new_features = [
    "foot_traffic",
    "competition_density",
    "has_marketing",
    "rent_to_revenue",
    "labor_cost_ratio",
]

print(f"Закрылись до трёх лет по всему train: {train[target].mean():.2%}")

print("\n=== Проверка пяти новых признаков ===")
print(
    train[new_features]
    .agg(["min", "median", "max"])
    .T.assign(missing=train[new_features].isna().sum())
    .round(4)
    .to_string()
)


def show_summary(title, groups):
    summary = train.groupby(groups, dropna=False, observed=True)[target].agg(
        restaurants="size",
        closed="sum",
        closed_share="mean",
    )
    summary["closed_share"] = summary["closed_share"].map(lambda share: f"{share:.2%}")
    print(f"\n=== {title} ===")
    print(summary.to_string())


# Категории сравниваем по исходным значениям.
for feature in ["cuisine", "price_tier", "has_marketing"]:
    show_summary(feature, train[feature])

# Числовые признаки делим на группы по квартилям обучающей выборки.
for feature in [
    "seats",
    "menu_size",
    "foot_traffic",
    "competition_density",
    "labor_cost_ratio",
]:
    groups = pd.qcut(train[feature], q=4, duplicates="drop")
    show_summary(feature, groups)

# Проверяем границу аренды из описания датасета.
rent_groups = pd.cut(
    train["rent_to_revenue"],
    bins=[float("-inf"), 0.10, float("inf")],
    labels=["До 10% включительно", "Выше 10%"],
)
show_summary("rent_to_revenue", rent_groups)