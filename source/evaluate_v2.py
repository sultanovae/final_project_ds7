from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score


project_root = Path(__file__).resolve().parents[1]
target = "closed_before_3y"

saved = joblib.load(project_root / "artifacts" / "model_v2.joblib")
low_cutoff, high_cutoff = saved["risk_cutoffs"]
test = pd.read_csv(project_root / "data" / "splits_v2" / "test.csv")

# Передаём модели ровно те семь столбцов, на которых она обучалась.
scores = saved["model"].predict_proba(test[saved["features"]])[:, 1]

risk_level = pd.cut(
    scores,
    bins=[float("-inf"), low_cutoff, high_cutoff, float("inf")],
    labels=["Низкий", "Средний", "Высокий"],
    right=False,
)

evaluated = test[["restaurant_id", target]].assign(risk_level=risk_level)

summary = evaluated.groupby("risk_level", observed=False)[target].agg(
    restaurants="size",
    closed="sum",
    closed_share="mean",
)
summary["closed_share_pct"] = summary.pop("closed_share") * 100
summary["group_share_pct"] = summary["restaurants"] / len(test) * 100
summary["share_of_all_closures_pct"] = (
    summary["closed"] / test[target].sum() * 100
)

print(f"AP на test: {average_precision_score(test[target], scores):.4f}")
print(f"Границы из validation: {low_cutoff:.6f} и {high_cutoff:.6f}")
print("\n=== Уровни риска на test ===")
print(summary.round(2).to_string())

# Причину закрытия подключаем после расчёта прогнозов.
reasons = pd.read_csv(
    project_root / "data" / "raw" / "restaurant_survival.csv",
    usecols=["restaurant_id", "failure_reason"],
)

closed = evaluated.loc[evaluated[target].eq(1)].merge(
    reasons,
    on="restaurant_id",
    how="left",
    validate="one_to_one",
)

closed["missed"] = closed["risk_level"].eq("Низкий")

error_summary = closed.groupby("failure_reason", dropna=False).agg(
    closed=("restaurant_id", "size"),
    missed=("missed", "sum"),
)
error_summary["missed_pct"] = (
    error_summary["missed"] / error_summary["closed"] * 100
)

print("\n=== Закрытия в низкой группе по причинам ===")
print(
    error_summary.sort_values(
        ["missed_pct", "missed"], ascending=False
    ).round(2).to_string()
)