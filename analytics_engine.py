#!/usr/bin/env python3
"""
Analytics Engine
Genera analytics_report.xlsx con múltiples hojas de análisis.
"""

import json
import re
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

INPUT_CSV = "videos_categorized.csv"
OUTPUT_XLSX = "analytics_report.xlsx"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data():
    """Carga y prepara el DataFrame."""
    df = pd.read_csv(INPUT_CSV)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["published_at"] = df["published_at"].dt.tz_localize(None)
    df["year"] = df["published_at"].dt.year
    df["month"] = df["published_at"].dt.month
    df["day_of_week"] = df["published_at"].dt.day_name()
    df["hour"] = df["published_at"].dt.hour

    # Engagement rate = (likes + comments) / views * 100
    df["engagement_rate"] = np.where(
        df["views"] > 0,
        (df["likes"] + df["comments"]) / df["views"] * 100,
        0
    )

    # Flag últimos 12 meses
    cutoff_12m = pd.Timestamp.now() - timedelta(days=365)
    df["last_12m"] = df["published_at"] >= cutoff_12m

    print(f"Videos cargados: {len(df)}")
    print(f"Rango: {df['year'].min()} - {df['year'].max()}")
    return df


def compute_group_metrics(group_df):
    """Calcula métricas promedio para un grupo."""
    n = len(group_df)
    if n == 0:
        return {}

    avg_views = group_df["views"].mean()
    avg_likes = group_df["likes"].mean()
    avg_comments = group_df["comments"].mean()
    avg_ctr = group_df.loc[group_df["ctr"] > 0, "ctr"].mean() if (group_df["ctr"] > 0).any() else 0
    avg_avd = group_df.loc[group_df["avg_view_duration_sec"] > 0, "avg_view_duration_sec"].mean() \
        if (group_df["avg_view_duration_sec"] > 0).any() else 0
    avg_avp = group_df.loc[group_df["avg_view_percentage"] > 0, "avg_view_percentage"].mean() \
        if (group_df["avg_view_percentage"] > 0).any() else 0
    avg_engagement = group_df["engagement_rate"].mean()
    total_views = group_df["views"].sum()
    avg_impressions = group_df.loc[group_df["impressions"] > 0, "impressions"].mean() \
        if (group_df["impressions"] > 0).any() else 0
    avg_subs = group_df.loc[group_df["subscribers_gained"] > 0, "subscribers_gained"].mean() \
        if (group_df["subscribers_gained"] > 0).any() else 0

    return {
        "num_videos": n,
        "total_views": int(total_views),
        "avg_views": round(avg_views, 0),
        "avg_likes": round(avg_likes, 0),
        "avg_comments": round(avg_comments, 0),
        "avg_ctr_pct": round(avg_ctr, 2),
        "avg_view_duration_sec": round(avg_avd, 1),
        "avg_view_percentage": round(avg_avp, 2),
        "avg_engagement_rate": round(avg_engagement, 3),
        "avg_impressions": round(avg_impressions, 0),
        "avg_subscribers_gained": round(avg_subs, 1),
    }


def add_composite_score(df_metrics):
    """Agrega score compuesto normalizado (0-100)."""
    if df_metrics.empty:
        return df_metrics

    cols_to_score = ["avg_views", "avg_ctr_pct", "avg_view_duration_sec",
                     "avg_engagement_rate"]
    available = [c for c in cols_to_score if c in df_metrics.columns]

    for col in available:
        col_min = df_metrics[col].min()
        col_max = df_metrics[col].max()
        col_range = col_max - col_min
        df_metrics[f"{col}_norm"] = (df_metrics[col] - col_min) / col_range if col_range > 0 else 0

    norm_cols = [f"{c}_norm" for c in available]
    weights = {"avg_views_norm": 0.3, "avg_ctr_pct_norm": 0.25,
               "avg_view_duration_sec_norm": 0.25, "avg_engagement_rate_norm": 0.2}

    df_metrics["composite_score"] = sum(
        df_metrics[c] * weights.get(c, 0.25) for c in norm_cols if c in df_metrics.columns
    ) * 100

    df_metrics["composite_score"] = df_metrics["composite_score"].round(1)
    df_metrics.drop(columns=[c for c in df_metrics.columns if c.endswith("_norm")],
                    inplace=True)

    return df_metrics.sort_values("composite_score", ascending=False)


# ---------------------------------------------------------------------------
# Hojas del reporte
# ---------------------------------------------------------------------------

def sheet_genre_performance(df):
    """a) Performance por género con score compuesto."""
    rows = []
    for genre, group in df.groupby("game_genre"):
        m = compute_group_metrics(group)
        m["genre"] = genre
        rows.append(m)
    result = pd.DataFrame(rows)
    if not result.empty:
        cols = ["genre"] + [c for c in result.columns if c != "genre"]
        result = result[cols]
        result = add_composite_score(result)
    return result


def sheet_format_performance(df):
    """b) Performance por formato."""
    rows = []
    for fmt, group in df.groupby("video_format"):
        m = compute_group_metrics(group)
        m["format"] = fmt
        rows.append(m)
    result = pd.DataFrame(rows)
    if not result.empty:
        cols = ["format"] + [c for c in result.columns if c != "format"]
        result = result[cols]
        result = add_composite_score(result)
    return result


def sheet_genre_by_year(df):
    """c) Performance por género filtrado por año (2020-2026) y últimos 12 meses."""
    rows = []

    # Por año
    for year in range(2020, 2027):
        year_df = df[df["year"] == year]
        if year_df.empty:
            continue
        for genre, group in year_df.groupby("game_genre"):
            m = compute_group_metrics(group)
            m["period"] = str(year)
            m["genre"] = genre
            rows.append(m)

    # Últimos 12 meses
    last12_df = df[df["last_12m"]]
    if not last12_df.empty:
        for genre, group in last12_df.groupby("game_genre"):
            m = compute_group_metrics(group)
            m["period"] = "Last 12 months"
            m["genre"] = genre
            rows.append(m)

    result = pd.DataFrame(rows)
    if not result.empty:
        cols = ["period", "genre"] + [c for c in result.columns if c not in ("period", "genre")]
        result = result[cols]
        result = result.sort_values(["period", "total_views"], ascending=[True, False])
    return result


def sheet_top100(df):
    """d) Deep dive de los 100 mejores videos."""
    top = df.nlargest(100, "views").copy()
    top["day_published"] = top["published_at"].dt.day_name()

    # Categorías de duración
    bins = [0, 300, 600, 900, 1800, 3600, float("inf")]
    labels = ["0-5min", "5-10min", "10-15min", "15-30min", "30-60min", "60+min"]
    top["duration_bucket"] = pd.cut(top["duration_seconds"], bins=bins, labels=labels)

    cols = [
        "video_id", "title", "game_name", "game_genre", "video_format", "visual_style",
        "published_at", "year", "day_published", "duration_seconds", "duration_bucket",
        "views", "likes", "comments", "engagement_rate",
        "impressions", "ctr", "avg_view_duration_sec", "avg_view_percentage",
        "subscribers_gained", "estimated_minutes_watched",
        "retention_30s", "retention_1min", "retention_50pct", "retention_70pct",
    ]
    available = [c for c in cols if c in top.columns]
    return top[available].reset_index(drop=True)


def sheet_title_keywords(df):
    """e) Análisis de palabras clave en títulos y su impacto."""
    keywords = [
        "Ultra Realistic", "4K", "HDR", "60FPS", "60fps", "60 FPS", "60 fps",
        "PS5", "PS4", "Xbox", "Ray Tracing",
        "Cinematic", "BEST", "NEVER", "AMAZING", "INCREDIBLE", "BEAUTIFUL",
        "STUNNING", "INSANE", "EPIC", "FREE ROAM",
        "First Person", "FIRST PERSON", "Next Gen", "NEXT GEN",
        "Stealth", "STEALTH", "Open World", "OPEN WORLD",
        "Full Match", "FULL MATCH", "Gameplay", "GAMEPLAY",
        "Realistic Graphics", "REALISTIC",
    ]
    # Deduplicar case-insensitive
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            unique_keywords.append(kw)

    rows = []
    for kw in unique_keywords:
        mask = df["title"].str.contains(kw, case=False, na=False)
        with_kw = df[mask]
        without_kw = df[~mask]

        if len(with_kw) < 3:
            continue

        m_with = compute_group_metrics(with_kw)
        m_without = compute_group_metrics(without_kw)

        views_lift = ((m_with["avg_views"] - m_without["avg_views"]) / m_without["avg_views"] * 100) \
            if m_without.get("avg_views", 0) > 0 else 0
        ctr_lift = (m_with["avg_ctr_pct"] - m_without["avg_ctr_pct"]) \
            if m_without.get("avg_ctr_pct") is not None else 0

        rows.append({
            "keyword": kw,
            "num_videos_with": m_with["num_videos"],
            "avg_views_with": m_with["avg_views"],
            "avg_views_without": m_without["avg_views"],
            "views_lift_pct": round(views_lift, 1),
            "avg_ctr_with": m_with["avg_ctr_pct"],
            "avg_ctr_without": m_without["avg_ctr_pct"],
            "ctr_lift": round(ctr_lift, 2),
            "avg_avd_with": m_with["avg_view_duration_sec"],
            "avg_avd_without": m_without["avg_view_duration_sec"],
            "avg_engagement_with": m_with["avg_engagement_rate"],
            "avg_engagement_without": m_without["avg_engagement_rate"],
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("views_lift_pct", ascending=False)
    return result


def sheet_timing(df):
    """f) Mejor día y hora para publicar."""
    rows = []

    # Por día de la semana
    for day, group in df.groupby("day_of_week"):
        m = compute_group_metrics(group)
        m["dimension"] = "day_of_week"
        m["value"] = day
        rows.append(m)

    # Por hora
    for hour, group in df.groupby("hour"):
        m = compute_group_metrics(group)
        m["dimension"] = "hour"
        m["value"] = f"{hour:02d}:00"
        rows.append(m)

    # Por día + hora (top combos)
    df["day_hour"] = df["day_of_week"] + " " + df["hour"].apply(lambda h: f"{h:02d}:00")
    for dh, group in df.groupby("day_hour"):
        if len(group) >= 3:
            m = compute_group_metrics(group)
            m["dimension"] = "day_hour_combo"
            m["value"] = dh
            rows.append(m)

    result = pd.DataFrame(rows)
    if not result.empty:
        cols = ["dimension", "value"] + [c for c in result.columns if c not in ("dimension", "value")]
        result = result[cols]
        result = result.sort_values(["dimension", "avg_views"], ascending=[True, False])
    return result


def sheet_funnel(df):
    """g) Conversión Impressions > CTR > Views > AVD > Engagement > Subs."""
    rows = []

    def add_funnel(label, subset):
        if subset.empty:
            return
        total_impressions = subset["impressions"].sum()
        total_views = subset["views"].sum()
        avg_ctr = subset.loc[subset["ctr"] > 0, "ctr"].mean() if (subset["ctr"] > 0).any() else 0
        avg_avd = subset.loc[subset["avg_view_duration_sec"] > 0, "avg_view_duration_sec"].mean() \
            if (subset["avg_view_duration_sec"] > 0).any() else 0
        avg_avp = subset.loc[subset["avg_view_percentage"] > 0, "avg_view_percentage"].mean() \
            if (subset["avg_view_percentage"] > 0).any() else 0
        avg_engagement = subset["engagement_rate"].mean()
        total_subs = subset["subscribers_gained"].sum()
        subs_per_1k_views = (total_subs / total_views * 1000) if total_views > 0 else 0

        rows.append({
            "segment": label,
            "num_videos": len(subset),
            "total_impressions": int(total_impressions),
            "avg_ctr_pct": round(avg_ctr, 2),
            "total_views": int(total_views),
            "impression_to_view_rate": round(total_views / total_impressions * 100, 2) if total_impressions > 0 else 0,
            "avg_view_duration_sec": round(avg_avd, 1),
            "avg_view_percentage": round(avg_avp, 2),
            "avg_engagement_rate": round(avg_engagement, 3),
            "total_subscribers_gained": int(total_subs),
            "subs_per_1k_views": round(subs_per_1k_views, 2),
        })

    add_funnel("ALL VIDEOS", df)
    valid = df[df["impressions"] > 0]
    if not valid.empty:
        add_funnel("With Impressions Data", valid)

    # Por género
    for genre, group in df.groupby("game_genre"):
        add_funnel(f"Genre: {genre}", group)

    # Por año (últimos 3)
    for year in sorted(df["year"].unique())[-3:]:
        add_funnel(f"Year: {year}", df[df["year"] == year])

    return pd.DataFrame(rows)


def sheet_hidden_gems(df):
    """h) Videos con CTR alto pero pocas impressions."""
    valid = df[(df["ctr"] > 0) & (df["impressions"] > 0)].copy()
    if valid.empty:
        return pd.DataFrame()

    ctr_median = valid["ctr"].median()
    impressions_median = valid["impressions"].median()

    gems = valid[
        (valid["ctr"] > ctr_median * 1.3) &
        (valid["impressions"] < impressions_median)
    ].copy()

    gems["ctr_vs_median"] = (gems["ctr"] / ctr_median * 100).round(1)
    gems["potential_views_at_median_impressions"] = (impressions_median * gems["ctr"] / 100).round(0)

    cols = [
        "video_id", "title", "game_name", "game_genre", "video_format",
        "published_at", "year", "views", "impressions", "ctr",
        "ctr_vs_median", "potential_views_at_median_impressions",
        "avg_view_duration_sec", "avg_view_percentage", "engagement_rate",
    ]
    available = [c for c in cols if c in gems.columns]
    return gems[available].sort_values("ctr", ascending=False).head(50).reset_index(drop=True)


def sheet_style_analysis(df):
    """i) Performance por estilo visual."""
    rows = []
    for style, group in df.groupby("visual_style"):
        m = compute_group_metrics(group)
        m["visual_style"] = style
        rows.append(m)
    result = pd.DataFrame(rows)
    if not result.empty:
        cols = ["visual_style"] + [c for c in result.columns if c != "visual_style"]
        result = result[cols]
        result = add_composite_score(result)
    return result


def sheet_cross_analysis(df):
    """j) Tabla cruzada género x formato x estilo visual."""
    rows = []
    for (genre, fmt, style), group in df.groupby(["game_genre", "video_format", "visual_style"]):
        if len(group) < 2:
            continue
        m = compute_group_metrics(group)
        m["genre"] = genre
        m["format"] = fmt
        m["visual_style"] = style
        rows.append(m)
    result = pd.DataFrame(rows)
    if not result.empty:
        cols = ["genre", "format", "visual_style"] + \
               [c for c in result.columns if c not in ("genre", "format", "visual_style")]
        result = result[cols]
        result = result.sort_values("total_views", ascending=False)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Analytics Engine")
    print("=" * 60)

    df = load_data()

    sheets = {}
    tasks = [
        ("genre_performance", "Género performance", sheet_genre_performance),
        ("format_performance", "Formato performance", sheet_format_performance),
        ("genre_by_year", "Género por año", sheet_genre_by_year),
        ("top100_analysis", "Top 100 videos", sheet_top100),
        ("title_keywords", "Keywords en títulos", sheet_title_keywords),
        ("timing_analysis", "Timing de publicación", sheet_timing),
        ("funnel", "Funnel de conversión", sheet_funnel),
        ("hidden_gems", "Hidden gems", sheet_hidden_gems),
        ("style_analysis", "Estilo visual", sheet_style_analysis),
        ("cross_analysis", "Análisis cruzado", sheet_cross_analysis),
    ]

    for sheet_name, desc, func in tqdm(tasks, desc="Generando hojas"):
        print(f"\n  -> {desc}...")
        result = func(df)
        sheets[sheet_name] = result
        print(f"     {len(result)} filas generadas")

    # Escribir Excel
    print(f"\nEscribiendo {OUTPUT_XLSX}...")
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        for sheet_name, data in sheets.items():
            if data is not None and not data.empty:
                data.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                pd.DataFrame({"info": ["No data available"]}).to_excel(
                    writer, sheet_name=sheet_name, index=False
                )

    print(f"\nReporte guardado: {OUTPUT_XLSX}")
    print(f"Hojas generadas: {len(sheets)}")

    # Resumen rápido
    print("\n--- Highlights ---")
    if "genre_performance" in sheets and not sheets["genre_performance"].empty:
        top_genre = sheets["genre_performance"].iloc[0]
        print(f"Mejor género (score): {top_genre.get('genre', 'N/A')} "
              f"({top_genre.get('composite_score', 0)})")

    if "format_performance" in sheets and not sheets["format_performance"].empty:
        top_fmt = sheets["format_performance"].iloc[0]
        print(f"Mejor formato (score): {top_fmt.get('format', 'N/A')} "
              f"({top_fmt.get('composite_score', 0)})")

    if "hidden_gems" in sheets and not sheets["hidden_gems"].empty:
        print(f"Hidden gems encontrados: {len(sheets['hidden_gems'])}")

    print("=" * 60)


if __name__ == "__main__":
    main()
