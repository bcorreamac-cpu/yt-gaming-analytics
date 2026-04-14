#!/usr/bin/env python3
"""
Views Predictor — Joy Of Gaming
Modelo profesional de predicción de views basado en 1,122 videos históricos.

Usa XGBoost con features temporales, de contenido y de título.
Incluye ajuste por tendencia temporal del canal (decaimiento post-2022).
"""

import json
import warnings
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

INPUT_CSV = "videos_categorized.csv"
MODEL_FILE = "views_model.pkl"
FEATURES_FILE = "model_features.json"

# ---------------------------------------------------------------------------
# Keywords que impactan views (validadas por análisis del canal)
# ---------------------------------------------------------------------------
TITLE_KEYWORDS = [
    'ultra realistic', '4k', 'hdr', '60fps', 'ps5', 'ps4',
    'insane', 'amazing', 'incredible', 'beautiful', 'stunning',
    'best', 'never', 'epic', 'brutal',
    'first person', 'stealth', 'free roam', 'open world',
    'full match', 'gameplay', 'realistic', 'ultra',
    'next gen', 'ray tracing', 'real life', 'unreal',
]


def load_and_prepare_data():
    """Carga el CSV y prepara features para el modelo."""
    df = pd.read_csv(INPUT_CSV)
    df['published_at'] = pd.to_datetime(df['published_at'], utc=True, errors='coerce').dt.tz_localize(None)

    # Limpiar videos template/test/streams
    mask = df['title'].str.contains(
        'Story/Clickbait|Template|GoPro|Transmisión|test |SE PARO|HandBreak|Opcion ',
        case=False, na=False
    )
    df = df[~mask].copy()

    # Filtrar videos con 0 views (no publicados o errores)
    df = df[df['views'] > 0].copy()

    # Target: log de views (distribución más normal)
    df['log_views'] = np.log1p(df['views'])

    # === FEATURES TEMPORALES ===
    df['year'] = df['published_at'].dt.year
    df['month'] = df['published_at'].dt.month
    df['day_of_week'] = df['published_at'].dt.dayofweek  # 0=Mon, 6=Sun
    df['hour'] = df['published_at'].dt.hour
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Días desde publicación (antigüedad = más tiempo para acumular views)
    df['days_since_pub'] = (datetime.now() - df['published_at']).dt.days

    # Tendencia del canal: factor de "alcance" por época
    # 2021 = pico, después cae. Esto es CRÍTICO para predicción precisa.
    channel_reach = {
        2017: 0.3, 2018: 0.3, 2019: 0.3,
        2020: 0.05, 2021: 1.0, 2022: 0.8,
        2023: 0.15, 2024: 0.03, 2025: 0.02, 2026: 0.01,
    }
    df['channel_reach_factor'] = df['year'].map(channel_reach).fillna(0.01)

    # === FEATURES DE CONTENIDO ===
    # Encodear categorías
    le_genre = LabelEncoder()
    le_format = LabelEncoder()
    le_style = LabelEncoder()
    le_game = LabelEncoder()

    df['genre_enc'] = le_genre.fit_transform(df['game_genre'].fillna('other'))
    df['format_enc'] = le_format.fit_transform(df['video_format_label'].fillna('Gameplay Puro'))
    df['style_enc'] = le_style.fit_transform(df['visual_style_label'].fillna('Gameplay Estándar'))

    # Para juegos: usar los top 30 + "other"
    game_counts = df['game_name'].value_counts()
    top_games = game_counts[game_counts >= 5].index.tolist()
    df['game_normalized'] = df['game_name'].apply(lambda x: x if x in top_games else '_OTHER_')
    le_game.fit(df['game_normalized'])
    df['game_enc'] = le_game.transform(df['game_normalized'])

    # Popularidad histórica del juego (avg views de otros videos del mismo juego)
    game_avg_views = df.groupby('game_name')['views'].transform('mean')
    df['game_historical_avg'] = np.log1p(game_avg_views)

    # Número de videos previos del mismo juego (saturación)
    df['game_video_count'] = df.groupby('game_name').cumcount()

    # === FEATURES DE DURACIÓN ===
    df['duration_minutes'] = df['duration_seconds'] / 60
    df['duration_bucket'] = pd.cut(
        df['duration_minutes'],
        bins=[0, 5, 10, 15, 20, 30, 60, 999],
        labels=[0, 1, 2, 3, 4, 5, 6]
    ).astype(float).fillna(3)

    # === FEATURES DE TÍTULO ===
    df['title_length'] = df['title'].str.len().fillna(0)
    df['title_word_count'] = df['title'].str.split().str.len().fillna(0)
    df['title_caps_ratio'] = df['title'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
    )
    df['title_has_pipe'] = df['title'].str.contains('\\|', na=False).astype(int)
    df['title_has_dash'] = df['title'].str.contains('[-–—]', na=False).astype(int)
    df['title_has_brackets'] = df['title'].str.contains('[\\[\\(]', na=False).astype(int)
    df['title_has_emoji'] = df['title'].apply(
        lambda x: 1 if any(ord(c) > 127 for c in str(x)) else 0
    )
    df['title_has_question'] = df['title'].str.contains('\\?', na=False).astype(int)
    df['title_has_exclamation'] = df['title'].str.contains('!', na=False).astype(int)

    # Keywords individuales
    for kw in TITLE_KEYWORDS:
        col = 'kw_' + kw.replace(' ', '_')
        df[col] = df['title'].str.contains(kw, case=False, na=False).astype(int)

    # Keyword count total
    df['keyword_count'] = sum(
        df['title'].str.contains(kw, case=False, na=False).astype(int)
        for kw in TITLE_KEYWORDS
    )

    # === FEATURES DE COMPETENCIA INTERNA ===
    # Videos publicados la misma semana (auto-competencia)
    df['week_key'] = df['published_at'].dt.isocalendar().week.astype(int)
    df['year_week'] = df['year'].astype(str) + '_' + df['week_key'].astype(str)
    week_counts = df.groupby('year_week')['video_id'].transform('count')
    df['videos_same_week'] = week_counts

    # Guardar encoders
    encoders = {
        'genre': le_genre, 'format': le_format,
        'style': le_style, 'game': le_game,
        'top_games': top_games,
    }

    return df, encoders


def get_feature_columns():
    """Lista de features para el modelo."""
    base = [
        'year', 'month', 'day_of_week', 'hour', 'is_weekend',
        'days_since_pub', 'channel_reach_factor',
        'genre_enc', 'format_enc', 'style_enc', 'game_enc',
        'game_historical_avg', 'game_video_count',
        'duration_minutes', 'duration_bucket',
        'title_length', 'title_word_count', 'title_caps_ratio',
        'title_has_pipe', 'title_has_dash', 'title_has_brackets',
        'title_has_emoji', 'title_has_question', 'title_has_exclamation',
        'keyword_count', 'videos_same_week',
    ]
    kw_cols = ['kw_' + kw.replace(' ', '_') for kw in TITLE_KEYWORDS]
    return base + kw_cols


def train_model(df):
    """Entrena XGBoost con validación temporal."""
    feature_cols = get_feature_columns()
    X = df[feature_cols].fillna(0)
    y = df['log_views']

    # Time Series Split (respeta orden temporal)
    tscv = TimeSeriesSplit(n_splits=5)

    # XGBoost con parámetros optimizados para este dataset
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation temporal
    print("Entrenando con validación temporal (5 folds)...")
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
    print(f"  R² por fold: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  R² promedio: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    mae_scores = -cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
    print(f"  MAE promedio (log): {mae_scores.mean():.3f}")

    # Entrenar modelo final con todos los datos
    model.fit(X, y)

    # Evaluar en todo el dataset
    y_pred = model.predict(X)
    y_real = np.expm1(y)
    y_pred_real = np.expm1(y_pred)

    # Métricas en escala real
    mae_real = mean_absolute_error(y_real, y_pred_real)
    r2_real = r2_score(y_real, y_pred_real)

    # Accuracy por rangos
    within_50pct = np.mean(np.abs(y_real - y_pred_real) / np.maximum(y_real, 1) < 0.5) * 100
    within_2x = np.mean((y_pred_real >= y_real * 0.5) & (y_pred_real <= y_real * 2.0)) * 100

    print(f"\n  Métricas en escala real:")
    print(f"  R²: {r2_real:.3f}")
    print(f"  MAE: {mae_real:,.0f} views")
    print(f"  Predicciones dentro del ±50%: {within_50pct:.1f}%")
    print(f"  Predicciones dentro del ×2: {within_2x:.1f}%")

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)

    print(f"\n  Top 15 features más importantes:")
    for _, row in importance.head(15).iterrows():
        print(f"    {row['importance']:.4f}  {row['feature']}")

    return model, importance


def predict_views(model, encoders, game, genre, video_format, visual_style,
                  title, duration_minutes, publish_day, publish_hour,
                  year=2026, month=None):
    """Predice views para un video hipotético."""
    if month is None:
        month = datetime.now().month

    # Preparar features
    features = {}

    # Temporales
    day_map = {'Lunes': 0, 'Martes': 1, 'Miércoles': 2, 'Jueves': 3,
               'Viernes': 4, 'Sábado': 5, 'Domingo': 6}
    features['year'] = year
    features['month'] = month
    features['day_of_week'] = day_map.get(publish_day, 0)
    features['hour'] = publish_hour
    features['is_weekend'] = 1 if features['day_of_week'] >= 5 else 0
    features['days_since_pub'] = 30  # Proyección a 30 días

    channel_reach = {
        2017: 0.3, 2020: 0.05, 2021: 1.0, 2022: 0.8,
        2023: 0.15, 2024: 0.03, 2025: 0.02, 2026: 0.01,
    }
    features['channel_reach_factor'] = channel_reach.get(year, 0.01)

    # Contenido
    try:
        features['genre_enc'] = encoders['genre'].transform([genre])[0]
    except (ValueError, KeyError):
        features['genre_enc'] = 0
    try:
        features['format_enc'] = encoders['format'].transform([video_format])[0]
    except (ValueError, KeyError):
        features['format_enc'] = 0
    try:
        features['style_enc'] = encoders['style'].transform([visual_style])[0]
    except (ValueError, KeyError):
        features['style_enc'] = 0

    game_norm = game if game in encoders['top_games'] else '_OTHER_'
    try:
        features['game_enc'] = encoders['game'].transform([game_norm])[0]
    except (ValueError, KeyError):
        features['game_enc'] = 0

    features['game_historical_avg'] = 10.0  # Default medio
    features['game_video_count'] = 0
    features['duration_minutes'] = duration_minutes
    features['duration_bucket'] = (
        0 if duration_minutes <= 5 else
        1 if duration_minutes <= 10 else
        2 if duration_minutes <= 15 else
        3 if duration_minutes <= 20 else
        4 if duration_minutes <= 30 else 5
    )

    # Título
    features['title_length'] = len(title)
    features['title_word_count'] = len(title.split())
    features['title_caps_ratio'] = sum(1 for c in title if c.isupper()) / max(len(title), 1)
    features['title_has_pipe'] = 1 if '|' in title else 0
    features['title_has_dash'] = 1 if any(c in title for c in '-–—') else 0
    features['title_has_brackets'] = 1 if any(c in title for c in '[(') else 0
    features['title_has_emoji'] = 1 if any(ord(c) > 127 for c in title) else 0
    features['title_has_question'] = 1 if '?' in title else 0
    features['title_has_exclamation'] = 1 if '!' in title else 0

    kw_count = 0
    for kw in TITLE_KEYWORDS:
        col = 'kw_' + kw.replace(' ', '_')
        val = 1 if kw.lower() in title.lower() else 0
        features[col] = val
        kw_count += val
    features['keyword_count'] = kw_count

    features['videos_same_week'] = 5  # Plan de 5 videos/semana

    # Crear DataFrame con el orden correcto
    feature_cols = get_feature_columns()
    X = pd.DataFrame([features])[feature_cols].fillna(0)

    # Predecir
    log_pred = model.predict(X)[0]
    pred_views = int(np.expm1(log_pred))

    # Rango de confianza (±1 std del MAE en log space)
    log_std = 1.2  # Aproximación basada en CV
    low = int(np.expm1(log_pred - log_std))
    high = int(np.expm1(log_pred + log_std))

    return pred_views, low, high


def main():
    print("=" * 60)
    print("  Views Predictor — Joy Of Gaming")
    print("=" * 60)

    # Cargar y preparar datos
    print("\nPreparando datos...")
    df, encoders = load_and_prepare_data()
    print(f"Videos para entrenamiento: {len(df)}")

    # Ordenar por fecha para validación temporal
    df = df.sort_values('published_at').reset_index(drop=True)

    # Entrenar
    print("\n" + "-" * 40)
    model, importance = train_model(df)

    # Guardar modelo
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'model': model, 'encoders': encoders}, f)
    print(f"\nModelo guardado: {MODEL_FILE}")

    # Guardar feature importance como JSON
    imp_dict = importance.head(20).to_dict('records')
    with open(FEATURES_FILE, 'w') as f:
        json.dump(imp_dict, f, indent=2)

    # === PREDICCIONES PARA SEMANA 1 DEL CALENDARIO ===
    print("\n" + "=" * 60)
    print("  PREDICCIONES — Semana 1 (Abr 20-26)")
    print("=" * 60)

    calendar_week1 = [
        {
            "game": "Black Myth: Wukong", "genre": "action_adventure",
            "format": "Showcase Visual", "style": "Ultra Realista",
            "title": "(PS5) Black Myth Wukong - INSANE Boss Fight | Ultra Realistic Graphics [4K HDR 60FPS]",
            "duration": 13, "day": "Lunes", "hour": 16,
        },
        {
            "game": "Hot Wheels Unleashed", "genre": "racing",
            "format": "Showcase Visual", "style": "Ultra Realista",
            "title": "(PS5) Hot Wheels Unleashed 2 - INSANE Kitchen Track | Ultra Realistic Graphics [4K HDR 60FPS]",
            "duration": 11, "day": "Martes", "hour": 16,
        },
        {
            "game": "Dying Light 2", "genre": "action_adventure",
            "format": "Primera Persona POV", "style": "Ultra Realista",
            "title": "(PS5) Dying Light 2 - INSANE Zombie Parkour | Ultra Realistic Graphics [4K HDR 60FPS]",
            "duration": 13, "day": "Miércoles", "hour": 19,
        },
        {
            "game": "Kingdom Come Deliverance", "genre": "rpg",
            "format": "Showcase Visual", "style": "Ultra Realista",
            "title": "(PS5) Kingdom Come Deliverance 2 - Ultra Realistic DUEL | Medieval Graphics [4K HDR 60FPS]",
            "duration": 13, "day": "Viernes", "hour": 18,
        },
        {
            "game": "Death Stranding", "genre": "action_adventure",
            "format": "Showcase Visual", "style": "Ultra Realista",
            "title": "(PS5) Death Stranding 2 - Is THIS Real Life? | Ultra Realistic Graphics [4K HDR 60FPS]",
            "duration": 13, "day": "Domingo", "hour": 16,
        },
    ]

    predictions = []
    for v in calendar_week1:
        pred, low, high = predict_views(
            model, encoders,
            game=v['game'], genre=v['genre'],
            video_format=v['format'], visual_style=v['style'],
            title=v['title'], duration_minutes=v['duration'],
            publish_day=v['day'], publish_hour=v['hour'],
        )
        predictions.append({
            'game': v['game'],
            'predicted_views': pred,
            'range_low': low,
            'range_high': high,
        })
        print(f"\n  {v['day']:10s} | {v['game']:30s}")
        print(f"  Predicción: {pred:>8,} views")
        print(f"  Rango:      {low:>8,} — {high:>8,}")

    # Guardar predicciones
    with open('predictions_week1.json', 'w') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
