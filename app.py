import json
import pickle
import numpy as np
import pandas as pd
import gradio as gr

# ── Загрузка модели и метаданных ─────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_meta.json", encoding="utf-8") as f:
    meta = json.load(f)

FEATURES = meta["features"]
MEANS    = meta["scaler"]["means"]
STDS     = meta["scaler"]["stds"]

# Коэффициенты денормализации из документации UCI датасета
# (исходные данные уже нормализованы в [0,1])
TEMP_MAX      = 41    # °C
ATEMP_MAX     = 50    # °C (ощущаемая)
HUM_MAX       = 100   # %
WINDSPEED_MAX = 67    # km/h


def predict(hr, mnth, yr_label, season, weekday_label,
            workingday, holiday, weather,
            temp_c, atemp_c, hum_pct, windspeed_kmh):

    # Перевод пользовательских значений → нормализованные [0,1]
    temp_norm      = temp_c      / TEMP_MAX
    atemp_norm     = atemp_c     / ATEMP_MAX
    hum_norm       = hum_pct     / HUM_MAX
    windspeed_norm = windspeed_kmh / WINDSPEED_MAX

    # Z-score (те же параметры, что в train.py)
    temp_z      = (temp_norm      - MEANS["temp"])      / STDS["temp"]
    atemp_z     = (atemp_norm     - MEANS["atemp"])     / STDS["atemp"]
    hum_z       = (hum_norm       - MEANS["hum"])       / STDS["hum"]
    windspeed_z = (windspeed_norm - MEANS["windspeed"]) / STDS["windspeed"]

    yr      = 0 if yr_label == "2011" else 1
    weekday = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"].index(weekday_label)

    # One-hot для season
    season_fall   = int(season == "Осень")
    season_spring = int(season == "Весна")
    season_summer = int(season == "Лето")
    season_winter = int(season == "Зима")

    # One-hot для weathersit
    w_clear      = int(weather == "Ясно")
    w_heavy_rain = int(weather == "Сильный дождь/снег")
    w_light_rain = int(weather == "Лёгкий дождь/снег")
    w_mist       = int(weather == "Туман")

    row = {
        "yr":                      yr,
        "mnth":                    mnth,
        "hr":                      hr,
        "holiday":                 int(holiday),
        "weekday":                 weekday,
        "workingday":              int(workingday),
        "temp":                    temp_z,
        "atemp":                   atemp_z,
        "hum":                     hum_z,
        "windspeed":               windspeed_z,
        "season_fall":             season_fall,
        "season_spring":           season_spring,
        "season_summer":           season_summer,
        "season_winter":           season_winter,
        "weathersit_clear":        w_clear,
        "weathersit_heavy_rain":   w_heavy_rain,
        "weathersit_light_rain":   w_light_rain,
        "weathersit_mist":         w_mist,
    }

    X = pd.DataFrame([row])[FEATURES]
    pred = int(np.round(model.predict(X)[0]))
    pred = max(0, pred)

    # Уровень загруженности
    if pred < 50:
        level = "Низкий"
    elif pred < 150:
        level = "Средний"
    elif pred < 300:
        level = "Высокий"
    else:
        level = "Очень высокий"

    mae = meta["metrics"]["MAE"]
    return (
        f"### Прогноз: {pred} велосипедов\n\n"
        f"**Уровень спроса:** {level}  \n"
        f"**Погрешность модели (MAE):** ±{mae:.0f} велосипедов"
    )


# ── Gradio интерфейс ──────────────────────────────────────────
with gr.Blocks(title="Прогноз аренды велосипедов") as demo:

    gr.Markdown("# Прогноз спроса на прокат велосипедов\nМодель: RandomForest · датасет Capital Bikeshare 2011–2012")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Время")
            hr       = gr.Slider(0, 23, value=9,  step=1,  label="Час суток")
            mnth     = gr.Slider(1, 12, value=6,  step=1,  label="Месяц")
            yr_label = gr.Radio(["2011", "2012"],  value="2012", label="Год")

        with gr.Column():
            gr.Markdown("### День")
            season       = gr.Dropdown(["Весна", "Лето", "Осень", "Зима"], value="Лето",  label="Сезон")
            weekday_label = gr.Dropdown(["Пн","Вт","Ср","Чт","Пт","Сб","Вс"], value="Пн", label="День недели")
            workingday   = gr.Checkbox(value=True,  label="Рабочий день")
            holiday      = gr.Checkbox(value=False, label="Праздник")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Погода")
            weather      = gr.Dropdown(
                ["Ясно", "Туман", "Лёгкий дождь/снег", "Сильный дождь/снег"],
                value="Ясно", label="Погодные условия"
            )
            temp_c       = gr.Slider(0, 41,  value=20, step=0.5, label="Температура (°C)")
            atemp_c      = gr.Slider(0, 50,  value=22, step=0.5, label="Ощущаемая температура (°C)")
            hum_pct      = gr.Slider(0, 100, value=60, step=1,   label="Влажность (%)")
            windspeed_kmh = gr.Slider(0, 67, value=10, step=0.5, label="Скорость ветра (км/ч)")

        with gr.Column():
            gr.Markdown("### Результат")
            output = gr.Markdown()
            btn    = gr.Button("Рассчитать прогноз", variant="primary")

    btn.click(
        fn=predict,
        inputs=[hr, mnth, yr_label, season, weekday_label,
                workingday, holiday, weather,
                temp_c, atemp_c, hum_pct, windspeed_kmh],
        outputs=output,
    )

    gr.Examples(
        examples=[
            [8,  9, "2012", "Осень",  "Пн", True,  False, "Ясно",               18, 20, 55, 12],
            [14, 7, "2012", "Лето",   "Сб", False, False, "Ясно",               28, 30, 40,  8],
            [22, 1, "2011", "Зима",   "Вт", True,  False, "Туман",               5,  3, 80, 15],
            [17, 5, "2012", "Весна",  "Пт", True,  False, "Лёгкий дождь/снег",  12, 13, 75, 20],
        ],
        inputs=[hr, mnth, yr_label, season, weekday_label,
                workingday, holiday, weather,
                temp_c, atemp_c, hum_pct, windspeed_kmh],
    )

if __name__ == "__main__":
    demo.launch()
