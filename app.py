from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


project_root = Path(__file__).resolve().parent
saved = joblib.load(project_root / "artifacts" / "model.joblib")
features = saved["features"]
threshold = saved["threshold"]

train = pd.read_csv(project_root / "data" / "splits" / "train.csv")

st.title("Проработает ли ресторан три года?")
st.write("Заполни характеристики заведения на старте.")

with st.form("restaurant"):
    cuisine = st.selectbox("Кухня", sorted(train["cuisine"].dropna().unique()))
    location = st.selectbox(
        "Тип локации", sorted(train["location_type"].dropna().unique())
    )
    franchise = st.checkbox("Заведение работает по франшизе")
    experience = st.number_input("Опыт владельца, лет", min_value=0.0, value=5.0)
    capital = st.number_input("Стартовый капитал, месяцев", min_value=0.0, value=3.0)
    price_tier = st.selectbox(
        "Ценовой уровень (код из датасета)", sorted(train["price_tier"].unique())
    )
    seats = st.number_input("Количество мест", min_value=1, value=48)
    menu_size = st.number_input("Количество позиций в меню", min_value=1, value=40)
    submitted = st.form_submit_button("Оценить")

if submitted:
    restaurant = pd.DataFrame([{
        "cuisine": cuisine,
        "location_type": location,
        "is_franchise": int(franchise),
        "owner_experience_years": experience,
        "starting_capital_months": capital,
        "price_tier": price_tier,
        "seats": seats,
        "menu_size": menu_size,
    }])

    score = saved["model"].predict_proba(restaurant[features])[0, 1]
    st.metric("Оценка модели для класса «не достигнет трёх лет»", f"{score:.3f}")
    st.caption(f"Порог сигнала: {threshold:.2f}")

    if score >= threshold:
        st.warning("Есть сигнал для дополнительной проверки.")
    else:
        st.success("Оценка ниже порога сигнала.")