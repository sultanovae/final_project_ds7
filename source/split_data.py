from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

project_root = Path(__file__).resolve().parents[1]
csv_path = project_root / "data" / "raw" / "restaurant_survival.csv"
output_dir = project_root / "data" / "splits"

df = pd.read_csv(csv_path)

# Признаки для первой модели. Уточним их смысл по описанию датасета.
features = [
    "cuisine",
    "location_type",
    "is_franchise",
    "owner_experience_years",
    "starting_capital_months",
    "price_tier",
    "seats",
    "menu_size",
]

data = df[features].copy()
data["closed_before_3y"] = df["survived_3y"].map({0: 1, 1: 0})

# Оставляем 60% на обучение, остальные 40% делим пополам.
train, remaining = train_test_split(
    data,
    test_size=0.40,
    random_state=42,
    stratify=data["closed_before_3y"],
)

validation, test = train_test_split(
    remaining,
    test_size=0.50,
    random_state=42,
    stratify=remaining["closed_before_3y"],
)

output_dir.mkdir(parents=True, exist_ok=True)

for name, part in [("train", train), ("validation", validation), ("test", test)]:
    part.to_csv(output_dir / f"{name}.csv", index=False)
    share_closed = part["closed_before_3y"].mean() * 100
    print(f"{name}: {len(part):,} строк, доля класса 1: {share_closed:.2f}%")