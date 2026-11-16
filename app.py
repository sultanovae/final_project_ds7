from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Ресторан · Проверка бизнес-плана",
    page_icon="🍽️",
    layout="wide",
)

# Цвета и оформление бизнес-панели.
st.markdown(
    """
    <style>
    .stApp { background: #f6f8f7; }
    .block-container { max-width: 1320px; padding-top: 2.2rem; }
    .dashboard-header {
        background: #FCB018;
        color: #ffffff;
        border-radius: 18px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.3rem;
    }
    .dashboard-header h1 {
        color: #ffffff;
        font-size: clamp(1.8rem, 4vw, 2.7rem);
        line-height: 1.15;
        margin: .35rem 0 .5rem;
        padding: 0;
    }
    .dashboard-header p { color: #FFFFFF; margin: 0; }
    .dashboard-label {
        color: #FFFFFF;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .09em;
    }
    .chart-row { margin: .65rem 0 .85rem; }
    .chart-label {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: .7rem;
        margin-bottom: .35rem;
        color: #20322c;
        font-size: .92rem;
    }
    .chart-track {
        background: #e6ece8;
        border-radius: 6px;
        height: 12px;
        overflow: hidden;
    }
    .chart-fill { height: 100%; border-radius: 6px; }
    .stApp,
.stApp :is(h1, h2, h3, p, label, button, input, textarea) {
    font-family: "Segoe UI", sans-serif;
}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="dashboard-header">
        <div class="dashboard-label">ОЦЕНКА ПЕРЕД ОТКРЫТИЕМ</div>
        <h1>Проверка бизнес-плана ресторана</h1>
        <p>Оценка риска закрытия в первые три года и сравнение сценариев.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model_v2.joblib"

if not MODEL_PATH.is_file():
    st.error("Модель не найдена. Сначала запусти source/train_v2.py.")
    st.stop()

saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]
low_cutoff, high_cutoff = saved["risk_cutoffs"]

LOCATIONS = {
    "downtown": "Центр города",
    "suburb_strip": "Торговая зона в пригороде",
    "mall_foodcourt": "Фудкорт в ТЦ",
    "standalone": "Отдельно стоящее заведение",
    "tourist_area": "Туристическая зона",
}

LEVEL_COLORS = {
    "Низкий": "#287553",
    "Средний": "#a76616",
    "Высокий": "#a74036",
}


def get_level(score):
    if score >= high_cutoff:
        return "Высокий"
    if score >= low_cutoff:
        return "Средний"
    return "Низкий"


def calculate_score(profile):
    data = pd.DataFrame([profile], columns=features)
    return float(model.predict_proba(data)[0, 1])


def show_bar(label, fraction, display_value, color):
    # Длина полосы соответствует значению на шкале от 0 до 1.
    width = max(0, min(100, fraction * 100))
    st.markdown(
        f"""
        <div class="chart-row">
            <div class="chart-label">
                <span>{label}</span><strong>{display_value}</strong>
            </div>
            <div class="chart-track">
                <div class="chart-fill"
                     style="width:{width:.2f}%;background:{color}"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if st.session_state.get("submitted_profile") is None:
    st.markdown(
        """
        <div style="
            text-align: center;
            background: #FFFFFF;
            color: #20322C;
            border-top: 4px solid #FCB018;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 20px;
            font-family: 'Segoe UI', sans-serif;
            font-size: 17px;
        ">
            <strong>Начни с параметров заведения</strong><br>
            Заполни форму ниже и нажми <strong>«Оценить бизнес-план»</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

inputs, dashboard = st.columns([1, 1.9], gap="large")

def reset_profile():
    st.session_state["input_location"] = "downtown"
    st.session_state["input_experience"] = 5.0
    st.session_state["input_capital"] = 3.0
    st.session_state["input_traffic"] = 5.0
    st.session_state["input_competition"] = 5.0
    st.session_state["input_franchise"] = False
    st.session_state["input_marketing"] = False
    st.session_state.pop("submitted_profile", None)


with inputs:
    with st.form("restaurant_profile"):
        st.subheader("Параметры заведения")

        location = st.selectbox(
            "Тип локации",
            list(LOCATIONS),
            format_func=LOCATIONS.get,
            key="input_location",
        )
        experience = st.number_input(
            "Опыт владельца, лет",
            min_value=0.0, value=5.0, step=0.5,
            key="input_experience",
        )
        capital = st.number_input(
            "Запас капитала, месяцев",
            min_value=0.0, value=3.0, step=0.5,
            key="input_capital",
        )
        traffic = st.slider(
            "Пешеходный поток, баллы",
            0.0, 10.0, 5.0, 0.1,
            key="input_traffic",
        )
        competition = st.slider(
            "Плотность конкурентов, баллы",
            0.0, 10.0, 5.0, 0.1,
            key="input_competition",
        )
        st.caption("Поток и конкуренция указаны в условных баллах датасета.")

        franchise = st.checkbox("Работа по франшизе", key="input_franchise")
        marketing = st.checkbox(
            "К открытию есть план продвижения",
            key="input_marketing",
        )

        calculate_column, reset_column = st.columns(2)
        with calculate_column:
            submitted = st.form_submit_button(
                "Оценить бизнес-план",
                type="primary",
                use_container_width=True,
            )
        with reset_column:
            st.form_submit_button(
                "Сбросить параметры",
                on_click=reset_profile,
                use_container_width=True,
            )

    if submitted:
        st.session_state["submitted_profile"] = {
            "location_type": location,
            "is_franchise": int(franchise),
            "owner_experience_years": experience,
            "starting_capital_months": capital,
            "foot_traffic": traffic,
            "competition_density": competition,
            "has_marketing": int(marketing),
        }
with dashboard:
    profile = st.session_state.get("submitted_profile")

    if profile is not None:
        score = calculate_score(profile)
        level = get_level(score)

        level_column, score_column = st.columns(2)
        level_column.metric("Уровень риска", level)
        score_column.metric("Оценка модели (логистическая регрессия)", f"{score:.3f}")
        st.caption("Оценка на шкале 0–1. Показатели групп приведены ниже.")

        if level == "Высокий":
            st.warning("Приоритетная проверка бизнес-плана перед вложениями.")
        elif level == "Средний":
            st.info("Сравни базовый и пессимистичный сценарии выручки.")
        else:
            st.success("Низкий уровень относительно заведений в датасете.")

        st.subheader("Что перепроверить")
        st.markdown(
            """
            - **Запас средств.** Посчитай, на сколько месяцев хватит денег,
              если выручка окажется ниже плана.
            - **Локацию.** Замерь поток в часы работы, сравни меню и цены
              ближайших конкурентов.
            - **Безубыточность.** Проверь, сколько заказов в день нужно
              для покрытия аренды, зарплат и других расходов.
            """
        )

        st.subheader("Как меняется оценка в разных сценариях")
        st.caption(
            "В каждом сценарии меняется один параметр. "
            "График показывает реакцию модели на ввод."
        )

        scenarios = [
            ("Запас капитала +2 месяца", {
                **profile,
                "starting_capital_months": profile["starting_capital_months"] + 2,
            }),
        ]
        if profile["foot_traffic"] < 10:
            scenarios.append((
                "Альтернативная точка с более высоким потоком",
                {**profile, "foot_traffic": min(10.0, profile["foot_traffic"] + 2)},
            ))
        if profile["competition_density"] > 0:
            scenarios.append((
                "Альтернативная точка с меньшей конкуренцией",
                {
                    **profile,
                    "competition_density": max(0.0, profile["competition_density"] - 2),
                },
            ))
        if not profile["has_marketing"]:
            scenarios.append((
                "Есть план продвижения",
                {**profile, "has_marketing": 1},
            ))

        show_bar("Текущий план", score, f"{score:.3f}", LEVEL_COLORS[level])
        for label, alternative in scenarios:
            alternative_score = calculate_score(alternative)
            alternative_level = get_level(alternative_score)
            difference = alternative_score - score
            show_bar(
                label,
                alternative_score,
                f"{alternative_score:.3f}  ({difference:+.3f})",
                LEVEL_COLORS[alternative_level],
            )

        st.caption(
            "Изменение оценки помогает выбрать, какой сценарий проверить "
            "в бизнес-плане. Фактический эффект этих действий на закрытия "
            "в проекте отдельно не измерялся."
        )

    st.divider()
    st.subheader("Что произошло с заведениями на тестовой выборке")
    st.caption("Доля заведений, которые не проработали три года, внутри каждого уровня.")

    # Результаты evaluate_v2.py для текущей версии модели.
    test_groups = [
        ("Низкий", 594, 2694),
        ("Средний", 2374, 4483),
        ("Высокий", 1479, 1823),
    ]
    for group, closed, total in test_groups:
        share = closed / total
        show_bar(
            group,
            share,
            f"{share:.2%}  ·  {closed:,} из {total:,}".replace(",", " "),
            LEVEL_COLORS[group],
        )

    st.caption(
        "Источник — тестовая выборка Kaggle, 9 000 заведений. "
        "Применимость к ресторанам Бишкека отдельно не проверялась."
    )

# .\.venv\Scripts\python.exe -m streamlit run app.py