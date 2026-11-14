## Итоговая оценка модели

Логистическая регрессия с C = 0.001 оценивалась на отдельной
тестовой выборке из 9 000 ресторанов. Порог 0.35 был выбран
по F1 на валидационной выборке.

На тесте Average Precision составила 0.6721, F1 — 0.6832,
accuracy — 57.76%, precision — 54.27%, recall — 92.20%.

Модель выявила 4 100 из 4 447 ресторанов, которые не достигли
трёх лет работы. Она пропустила 347 таких ресторанов и подала
3 455 ложных сигналов. Сигнал можно использовать как повод
для дополнительной проверки заведения.

## Как работает проект

Модель по восьми характеристикам ресторана оценивает риск того,
что он не проработает три года. Если оценка достигает порога 0.35,
приложение показывает сигнал для дополнительной проверки.

- `source/eda.py` изучает исходный CSV и обучающую выборку.
- `source/split_data.py` делит данные на train, validation и test
  в пропорции 60 / 20 / 20.
- `source/train.py` обучает модели на train, сравнивает их на
  validation и сохраняет выбранную модель и порог в `artifacts/model.joblib`.
- `source/evaluate.py` считает итоговые метрики на test.
- `app.py` показывает форму в Streamlit и загружает сохранённую модель.

## Локальный запуск на Windows

Скачайте датасет Restaurant Survival с Kaggle. Положите файл
`restaurant_survival.csv` в папку `data/raw/`.

https://www.kaggle.com/datasets/sergionefedov/restaurant-survival-why-they-really-close

Откройте папку проекта в VS Code. Выполните в PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\source\split_data.py
.\.venv\Scripts\python.exe .\source\eda.py
.\.venv\Scripts\python.exe .\source\train.py
.\.venv\Scripts\python.exe .\source\evaluate.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Откройте `Local URL` из терминала, заполните форму и нажмите «Оценить».
Модель обучена на данных Kaggle. Её качество для ресторанов Бишкека
отдельно не проверялось.