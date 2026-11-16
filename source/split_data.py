from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


project_root = Path(__file__).resolve().parents[1]
csv_path = project_root / "data" / "raw" / "restaurant_survival.csv"
old_dir = project_root / "data" / "splits"
new_dir = project_root / "data" / "splits_v2"

target = "closed_before_3y"

# A — исходные восемь признаков.
base_features = [
    "cuisine",
    "location_type",
    "is_franchise",
    "owner_experience_years",
    "starting_capital_months",
    "price_tier",
    "seats",
    "menu_size",
]

# B — добавляем сведения о месте и продвижении.
location_and_marketing = [
    "foot_traffic",
    "competition_density",
    "has_marketing",
]

# C — дополнительно проверяем финансовые показатели.
financial_features = [
    "rent_to_revenue",
    "labor_cost_ratio",
]

df = pd.read_csv(csv_path)

if df["restaurant_id"].isna().any() or not df["restaurant_id"].is_unique:
    raise ValueError("Проверь restaurant_id: нужны уникальные ID без пропусков.")

labels = df["survived_3y"].map({0: 1, 1: 0})
if labels.isna().any():
    raise ValueError("В survived_3y должны быть только 0 и 1.")

columns = [
    "restaurant_id",
    *base_features,
    *location_and_marketing,
    *financial_features,
]
data = df[columns].copy()
data[target] = labels.astype(int)

train, remaining = train_test_split(
    data,
    test_size=0.40,
    random_state=42,
    stratify=data[target],
)

validation, test = train_test_split(
    remaining,
    test_size=0.50,
    random_state=42,
    stratify=remaining[target],
)

splits = {
    "train": train,
    "validation": validation,
    "test": test,
}

# В старых CSV нет ID, поэтому сверяем исходные признаки и метку построчно.
original_columns = [*base_features, target]

for name, part in splits.items():
    old_path = old_dir / f"{name}.csv"
    if not old_path.is_file():
        continue
    old = pd.read_csv(old_path)
    reconstructed = part[original_columns].reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(
            old[original_columns],
            reconstructed,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise ValueError(
            f"{name}: старое и новое разбиения не совпали. "
            "Проверь, что исходный CSV и порядок его строк не менялись."
        ) from error

    print(f"{name}: совпадение со старой выборкой проверено")

new_dir.mkdir(parents=True, exist_ok=True)

for name, part in splits.items():
    part.to_csv(new_dir / f"{name}.csv", index=False)
    closed_share = part[target].mean() * 100
    print(f"{name}: {len(part):,} строк, доля класса 1: {closed_share:.2f}%")