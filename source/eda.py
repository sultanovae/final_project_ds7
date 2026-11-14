from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]
csv_path = project_root / "data" / "raw" / "restaurant_survival.csv"

df = pd.read_csv(csv_path)

print(f"Размер таблицы: {df.shape[0]} строк, {df.shape[1]} столбцов")
print("\nСтолбцы и типы данных:")
print(df.dtypes.to_string())

target = df["survived_3y"]

print("\nКоличество ресторанов по каждой метке:")
print(target.value_counts(dropna=False).to_string())

print("\nДоля каждой метки, %:")
print(target.value_counts(normalize=True, dropna=False).mul(100).round(2).to_string())

print("\nКак метка связана с количеством лет работы:")
print(pd.crosstab(df["years_survived"], target, dropna=False).to_string())

print("\nПроверка строк и идентификаторов:")
print("Пропуски в restaurant_id:", df["restaurant_id"].isna().sum())
print("Уникальных restaurant_id:", df["restaurant_id"].nunique())
print("Полных дубликатов строк:", df.duplicated().sum())

train_path = project_root / "data" / "splits" / "train.csv"
train = pd.read_csv(train_path)
target = "closed_before_3y"

print("\n=== Подробный EDA на train ===")

print("\nПропуски по столбцам:")
print(train.isna().sum().to_string())

numeric_columns = [
    "owner_experience_years",
    "starting_capital_months",
    "seats",
    "menu_size",
]

print("\nСводка по числовым признакам:")
print(train[numeric_columns].describe().round(2).to_string())

for column in ["is_franchise", "location_type"]:
    summary = train.groupby(column, dropna=False)[target].agg(
        restaurants="size",
        closed_share="mean",
    )
    summary["closed_share"] = (summary["closed_share"] * 100).round(2)

    print(f"\nДоля класса 1 по {column}, %:")
    print(summary.to_string())

train[numeric_columns].hist(bins=25, figsize=(10, 7))
plt.suptitle("Распределения признаков на train")
plt.tight_layout()
plt.show()