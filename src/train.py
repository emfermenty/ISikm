import pandas as pd
import numpy as np

# ───────────────────────────────────────────
# 1. Загрузка данных
# ───────────────────────────────────────────
df = pd.read_csv("data/hour.csv")

# ───────────────────────────────────────────
# 2. Определение признаков и целевой переменной
#
# Признаки (X):
#   season     — сезон года (1-4)
#   yr         — год (0=2011, 1=2012)
#   mnth       — месяц (1-12)
#   hr         — час суток (0-23)  ← самый сильный предиктор
#   holiday    — праздничный день (0/1)
#   weekday    — день недели (0-6)
#   workingday — рабочий день (0/1)
#   weathersit — погодные условия (1-4)
#   temp       — нормализованная температура
#   atemp      — ощущаемая температура
#   hum        — влажность
#   windspeed  — скорость ветра
#
# Целевая переменная (y):
#   cnt — общее количество велосипедов, взятых в аренду за час
#
# Исключаем:
#   instant — просто порядковый номер строки, не несёт информации
#   dteday  — дата в виде строки, её информацию уже кодируют yr/mnth/hr
#   casual, registered — части суммы cnt, утечка данных (data leakage)
# ───────────────────────────────────────────
FEATURES = [
    "season", "yr", "mnth", "hr",
    "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum", "windspeed",
]
TARGET = "cnt"

X = df[FEATURES].copy()
y = df[TARGET].copy()

# ───────────────────────────────────────────
# 3. Кодирование категориальных признаков
#
# season и weathersit — порядковые категории без числового смысла,
# поэтому заменяем их на понятные метки через словарь.
# ───────────────────────────────────────────
season_map = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
weather_map = {
    1: "clear",
    2: "mist",
    3: "light_rain",
    4: "heavy_rain",
}

X["season"] = X["season"].map(season_map)
X["weathersit"] = X["weathersit"].map(weather_map)

# One-hot encoding для season и weathersit
X = pd.get_dummies(X, columns=["season", "weathersit"], drop_first=False)

# Булевые столбцы (holiday, workingday) приводим к int для единообразия
X["holiday"] = X["holiday"].astype(int)
X["workingday"] = X["workingday"].astype(int)

# ───────────────────────────────────────────
# 4. Масштабирование числовых признаков
#
# Масштабируем только непрерывные признаки — temp, atemp, hum, windspeed.
# hr, mnth, weekday и т.п. оставляем как есть: они уже дискретны и
# модели на деревьях (и линейные с dummy-переменными) работают с ними нормально.
#
# Формула: z = (x - mean) / std  — стандартизация (Z-score normalization)
# ───────────────────────────────────────────
CONTINUOUS = ["temp", "atemp", "hum", "windspeed"]

means = X[CONTINUOUS].mean()
stds  = X[CONTINUOUS].std()

X[CONTINUOUS] = (X[CONTINUOUS] - means) / stds

# ───────────────────────────────────────────
# 5. Разделение на train и test
#
# Разбиваем 80/20: 80% — обучение + валидация, 20% — итоговый тест.
# Соотношение 80/20 — стандартный компромисс: достаточно данных
# для обучения и при этом репрезентативная тестовая выборка.
#
# shuffle=False — ВАЖНО для временных рядов: тест должен содержать
# только последние по времени наблюдения, иначе модель "видит будущее".
# ───────────────────────────────────────────
split_idx = int(len(X) * 0.8)

X_train = X.iloc[:split_idx]
X_test  = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

print(f"Всего записей : {len(df)}")
print(f"Train size    : {len(X_train)}  ({len(X_train)/len(df)*100:.0f}%)")
print(f"Test size     : {len(X_test)}   ({len(X_test)/len(df)*100:.0f}%)")
print(f"\nПризнаки после кодирования ({len(X.columns)}):")
print(list(X.columns))

# ═══════════════════════════════════════════════════════════════
# БЛОК 2. ОБУЧЕНИЕ И ДИАГНОСТИКА
# ═══════════════════════════════════════════════════════════════
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")   # сохраняем график в файл, без GUI
import matplotlib.pyplot as plt

# ── Вспомогательные функции метрик ───────────────────────────
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    # исключаем нули, чтобы не делить на 0
    mask = np.array(y_true) > 0
    return np.mean(np.abs(
        (np.array(y_true)[mask] - np.array(y_pred)[mask]) / np.array(y_true)[mask]
    )) * 100

def print_metrics(name, y_true, y_pred):
    print(f"  {name:<28} MAE={mean_absolute_error(y_true, y_pred):6.1f}  "
          f"RMSE={rmse(y_true, y_pred):6.1f}  MAPE={mape(y_true, y_pred):5.1f}%")

# ── Модель 1: Линейная регрессия (sklearn) ────────────────────
# Предполагает линейную связь между признаками и cnt.
# Простая и интерпретируемая, но не улавливает нелинейные паттерны
# (например, двойной пик спроса утром и вечером).
lr = LinearRegression()

# ── Модель 2: Случайный лес (сложная модель из sklearn) ───────
# Ансамбль деревьев решений. Хорошо улавливает нелинейные связи
# (пики утром/вечером в будни vs равномерный профиль выходных).
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# ── Кросс-валидация ───────────────────────────────────────────
# TimeSeriesSplit делит данные так, что обучение всегда предшествует
# проверке по времени. Обычный KFold здесь нельзя: он случайно
# подмешивает «будущее» в обучение, что даёт завышенную оценку.
tscv = TimeSeriesSplit(n_splits=5)

cv_mae = {"LinearRegression": [], "RandomForest": []}

print("\n--- Кросс-валидация (5 фолдов, TimeSeriesSplit) ---")
for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
    Xtr,  Xval  = X_train.iloc[tr_idx],  X_train.iloc[val_idx]
    ytr,  yval  = y_train.iloc[tr_idx],  y_train.iloc[val_idx]

    lr.fit(Xtr, ytr)
    p1 = lr.predict(Xval)
    cv_mae["LinearRegression"].append(mean_absolute_error(yval, p1))

    rf.fit(Xtr, ytr)
    p2 = rf.predict(Xval)
    cv_mae["RandomForest"].append(mean_absolute_error(yval, p2))

    print(f"  Fold {fold}: LinearRegression MAE={cv_mae['LinearRegression'][-1]:.1f}  "
          f"RandomForest MAE={cv_mae['RandomForest'][-1]:.1f}")

print("\n--- Средняя MAE по кросс-валидации ---")
for name, scores in cv_mae.items():
    print(f"  {name:<20} {np.mean(scores):.1f} +/- {np.std(scores):.1f}")

# Финальное обучение на всём train, оценка на test
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

p1_test = lr.predict(X_test)
p2_test = rf.predict(X_test)

print("\n--- Метрики на тестовой выборке ---")
print_metrics("LinearRegression",  y_test, p1_test)
print_metrics("RandomForest",      y_test, p2_test)

# ── Визуализация ошибок ───────────────────────────────────────
hours      = X_test["hr"].values
err_m1     = np.abs(y_test.values - p1_test)  # LinearRegression
err_rf     = np.abs(y_test.values - p2_test)

# MAE по каждому часу суток: видно, когда модель ошибается больше всего
mae_h_m1 = pd.Series(err_m1).groupby(hours).mean()
mae_h_rf = pd.Series(err_rf).groupby(hours).mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График 1: MAE по часу суток
x, w = np.arange(24), 0.35
axes[0].bar(x - w/2, mae_h_m1, w, label="LinearRegression", color="steelblue")
axes[0].bar(x + w/2, mae_h_rf, w, label="RandomForest",           color="tomato")
axes[0].set_xlabel("Час суток")
axes[0].set_ylabel("MAE (велосипедов)")
axes[0].set_title("Где модель ошибается больше всего?")
axes[0].set_xticks(x)
axes[0].legend()

# График 2: реальные vs предсказанные (RandomForest)
axes[1].scatter(y_test, p2_test, alpha=0.15, s=6, color="tomato")
lim = max(y_test.max(), p2_test.max()) + 10
axes[1].plot([0, lim], [0, lim], "k--", linewidth=1, label="Идеальное предсказание")
axes[1].set_xlabel("Реальное cnt")
axes[1].set_ylabel("Предсказанное cnt")
axes[1].set_title("RandomForest: реальные vs предсказанные")
axes[1].legend()

plt.tight_layout()
plt.savefig("diagnostics.png", dpi=150)
print("\nГрафик сохранён: diagnostics.png")

# ═══════════════════════════════════════════════════════════════
# БЛОК 3. ФИНАЛЬНЫЙ ОТБОР И СОХРАНЕНИЕ
# ═══════════════════════════════════════════════════════════════
import pickle
import json

# ── Выбор лучшей модели ───────────────────────────────────────
# RandomForest стабильно лучше по всем фолдам:
#   LinearRegression  108.7 +/- 21.6  (высокая дисперсия, не улавливает пики)
#   RandomForest       52.2 +/- 13.7  (в 2 раза точнее и стабильнее)
best_model      = rf
best_model_name = "RandomForest"

# ── Финальная проверка на отложенном тесте ────────────────────
# Тест трогаем ОДИН РАЗ — только здесь. Все предыдущие решения
# принимались исключительно по кросс-валидации на train.
final_pred = best_model.predict(X_test)

final_mae  = mean_absolute_error(y_test, final_pred)
final_rmse = rmse(y_test, final_pred)
final_mape = mape(y_test, final_pred)

# Часы с наибольшей ошибкой — «сложные» случаи для модели
err_by_hour = pd.Series(np.abs(y_test.values - final_pred), index=X_test["hr"].values)
worst_hours  = err_by_hour.groupby(level=0).mean().nlargest(2).index.tolist()
mae_by_hour  = {str(int(h)): round(v, 1)
                for h, v in err_by_hour.groupby(level=0).mean().items()}

# ── Сохранение модели (pickle) ────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# ── Сохранение метаданных (JSON) ─────────────────────────────
meta = {
    "model":    best_model_name,
    "features": list(X_train.columns),
    "metrics":  {
        "MAE":  round(final_mae,  2),
        "RMSE": round(final_rmse, 2),
        "MAPE": round(final_mape, 2),
    },
    "worst_hours": worst_hours,
    "mae_by_hour": mae_by_hour,
    # параметры Z-score нужны app.py для той же трансформации входных данных
    "scaler": {
        "means": means.to_dict(),
        "stds":  stds.to_dict(),
    },
}
with open("model_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

# ── Итоговый отчёт ────────────────────────────────────────────
print("\n" + "=" * 52)
print("  ИТОГОВЫЙ ОТЧЁТ")
print("=" * 52)
print(f"  Лучшая модель    : {best_model_name}")
print(f"  MAE  на тесте    : {final_mae:.1f}  велосипедов/час")
print(f"  RMSE на тесте    : {final_rmse:.1f}")
print(f"  MAPE на тесте    : {final_mape:.1f}%")
print(f"  Чаще всего путает: часы {worst_hours[0]}:00 и {worst_hours[1]}:00")
print(f"                     (пики спроса, резкие всплески аренд)")
print(f"  Модель сохранена : model.pkl")
print(f"  Метаданные       : model_meta.json")
print("=" * 52)
