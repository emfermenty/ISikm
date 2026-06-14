import gradio as gr
import src.app as app


# Тест 1: функция предсказания загружается и не падает на корректном примере
def test_predict_runs_without_error():
    result = app.predict(
        hr=9, mnth=6, yr_label="2012", season="Лето",
        weekday_label="Пн", workingday=True, holiday=False,
        weather="Ясно", temp_c=20, atemp_c=22, hum_pct=60, windspeed_kmh=10,
    )
    assert result is not None


# Тест 2: функция возвращает строку и содержит метку "велосипедов"
def test_predict_returns_string_with_count():
    result = app.predict(
        hr=9, mnth=6, yr_label="2012", season="Лето",
        weekday_label="Пн", workingday=True, holiday=False,
        weather="Ясно", temp_c=20, atemp_c=22, hum_pct=60, windspeed_kmh=10,
    )
    assert isinstance(result, str)
    assert "велосипедов" in result


# Тест 3: объект Gradio-приложения создаётся без ошибок
def test_demo_is_gradio_blocks():
    assert app.demo is not None
    assert isinstance(app.demo, gr.Blocks)
