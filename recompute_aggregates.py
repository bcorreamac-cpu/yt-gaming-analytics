#!/usr/bin/env python3
"""recompute_aggregates.py — Regenerate stale analytical aggregates in
strategic_data.json from videos_categorized.csv.

Designed to run AFTER audit_validator.py inside the daily workflow.
Preserves fields owned by other scripts (keywords, timing_*, retention_*,
genre_all, games_by_genre, top_games, calendar, model_info, forecast,
weekly_audit, game_progress_config, traffic_sources, channel_activity).

Why: many analytical fields were generated manually months ago and never
refreshed. Daily audit only updates a handful, leaving the dashboard with
stale numbers (e.g. performance_2026.videos missing the latest entries,
quarterly counts wrong, top100 not seeing post-2023 hits).

Outputs are schema-faithful to the existing strategic_data.json so the
dashboard JS keeps working without changes.
"""
from __future__ import annotations

import json
import math
import unicodedata
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "videos_categorized.csv"
JSON_PATH = ROOT / "strategic_data.json"
MANUAL_HEATMAP_PATH = ROOT / "manual_audience_heatmap.json"
MANUAL_DEMOGRAPHICS_PATH = ROOT / "manual_audience_demographics.json"
TODAY = date.today()
CURRENT_YEAR = TODAY.year

MONTH_ABBR_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    # Parse published_at as UTC then convert to Chile local time so all
    # year/month/quarter/day buckets are consistent with the rest of the
    # codebase (audit_validator.py uses UTC for its day/hour buckets but
    # the analytical aggregates here are always reported in Chile time —
    # the user's reference frame).
    pub = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    pub = pub.dt.tz_convert("America/Santiago").dt.tz_localize(None)
    df["_pub"] = pub
    df = df.dropna(subset=["_pub"]).copy()
    df["year"] = df["_pub"].dt.year
    df["month"] = df["_pub"].dt.month
    df["quarter"] = df["_pub"].dt.quarter
    df["week"] = df["_pub"].dt.isocalendar().week.astype(int)
    df["date"] = df["_pub"].dt.date
    # Ensure numerics
    for col in (
        "views", "likes", "comments", "duration_seconds",
        "avg_view_duration_sec", "avg_view_percentage",
        "subscribers_gained", "estimated_minutes_watched",
        "retention_30s", "retention_1min", "retention_50pct", "retention_70pct",
        "engagement_rate",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filtro defensivo: excluir Shorts (<=60s) y videos con #Shorts en el título.
    # data_collector.py ya filtra al ingest, pero por si el CSV tiene rezagos
    # de runs anteriores con el filtro viejo (<60), aplicamos el filtro aquí.
    before = len(df)
    if "duration_seconds" in df.columns:
        df = df[df["duration_seconds"] > 60].copy()
    if "title" in df.columns:
        df = df[~df["title"].astype(str).str.contains(r"#[Ss]horts", regex=True, na=False)].copy()
    excluded = before - len(df)
    if excluded:
        print(f"  Excluded {excluded} Shorts videos from analytics")
    return df


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _safe_round(v, n=1):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return 0
    return round(float(v), n)


def _safe_int(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return 0
    return int(v)


def parse_traffic(s):
    """traffic_sources column → dict (returns {} on null/garbage)."""
    if s is None:
        return {}
    if isinstance(s, float) and math.isnan(s):
        return {}
    if not isinstance(s, str) or not s.strip():
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _ts_views(ts: dict, key: str) -> int:
    v = ts.get(key)
    if isinstance(v, dict):
        return int(v.get("views", 0) or 0)
    if isinstance(v, (int, float)):
        return int(v)
    return 0


def quarter_dates(year: int, quarter: int):
    m_start = (quarter - 1) * 3 + 1
    start = date(year, m_start, 1)
    if quarter == 4:
        end = date(year, 12, 31)
    else:
        next_start = date(year, m_start + 3, 1)
        end = next_start - timedelta(days=1)
    return start, end


def is_current_quarter(year: int, quarter: int, today: date) -> bool:
    return year == today.year and ((today.month - 1) // 3 + 1) == quarter


# --------------------------------------------------------------------------
# Computations
# --------------------------------------------------------------------------

def compute_overview(df: pd.DataFrame) -> dict:
    return {
        "total_views": _safe_int(df["views"].sum()),
        "total_videos": int(len(df)),
        "total_likes": _safe_int(df["likes"].sum()),
        "total_comments": _safe_int(df["comments"].sum()),
        "total_subs": _safe_int(df["subscribers_gained"].sum()),
        "total_minutes": _safe_int(df["estimated_minutes_watched"].sum()),
        "avg_avd": _safe_round(df.loc[df["avg_view_duration_sec"] > 0, "avg_view_duration_sec"].mean(), 1),
        "avg_engagement": _safe_round(df["engagement_rate"].mean(), 3),
        "avg_views_per_video": _safe_round(df["views"].mean(), 0),
        "avg_retention_30s": _safe_round(df.loc[df["retention_30s"] > 0, "retention_30s"].mean(), 1),
    }


def compute_yearly_kpis(df: pd.DataFrame) -> dict:
    out = {}
    for y, grp in df.groupby("year"):
        if y < 2021:
            continue  # historical schema starts at 2021
        total_views = _safe_int(grp["views"].sum())
        total_subs = _safe_int(grp["subscribers_gained"].sum())
        ret_grp = grp[grp["retention_30s"] > 0]
        out[str(int(y))] = {
            "total_views": total_views,
            "total_videos": int(len(grp)),
            "avg_views_per_video": _safe_round(grp["views"].mean(), 0),
            "avg_avd": _safe_round(grp.loc[grp["avg_view_duration_sec"] > 0, "avg_view_duration_sec"].mean(), 1),
            "avg_engagement": _safe_round(grp["engagement_rate"].mean(), 3),
            "total_subs": total_subs,
            "total_minutes": _safe_int(grp["estimated_minutes_watched"].sum()),
            "subs_per_1k": _safe_round(total_subs / max(total_views, 1) * 1000, 2),
            "avg_retention_30s": (
                _safe_round(ret_grp["retention_30s"].mean(), 1) if len(ret_grp) else None
            ),
        }
    return out


def compute_yearly(df: pd.DataFrame) -> list:
    rows = []
    for y, grp in df.groupby("year"):
        if y < 2021:
            continue
        rows.append({
            "year": int(y),
            "total_views": _safe_int(grp["views"].sum()),
            "num_videos": int(len(grp)),
            "avg_views": _safe_round(grp["views"].mean(), 2),
            "avg_avd": _safe_round(grp.loc[grp["avg_view_duration_sec"] > 0, "avg_view_duration_sec"].mean(), 2),
            "avg_engagement": _safe_round(grp["engagement_rate"].mean(), 2),
            "avg_subs": _safe_round(grp["subscribers_gained"].mean(), 2),
        })
    return sorted(rows, key=lambda r: r["year"])


def compute_monthly_by_year(df: pd.DataFrame) -> list:
    rows = []
    for (y, m), grp in df.groupby(["year", "month"]):
        if y < 2021:
            continue
        ret = grp[grp["retention_30s"] > 0]
        rows.append({
            "year": int(y),
            "month": int(m),
            "videos": int(len(grp)),
            "views": _safe_int(grp["views"].sum()),
            "avg_views": _safe_round(grp["views"].mean(), 2),
            "avg_avd": _safe_round(grp.loc[grp["avg_view_duration_sec"] > 0, "avg_view_duration_sec"].mean(), 2),
            "avg_engagement": _safe_round(grp["engagement_rate"].mean(), 2),
            "avg_retention_30s": _safe_round(ret["retention_30s"].mean(), 2) if len(ret) else None,
        })
    return sorted(rows, key=lambda r: (r["year"], r["month"]))


def compute_genre_by_year(df: pd.DataFrame) -> dict:
    out = {}
    for y, grp in df.groupby("year"):
        rows = []
        for g, gg in grp.groupby("game_genre"):
            ret = gg[gg["retention_30s"] > 0]
            rows.append({
                "genre": g,
                "num_videos": int(len(gg)),
                "total_views": _safe_int(gg["views"].sum()),
                "avg_views": _safe_round(gg["views"].mean(), 0),
                "avg_avd": _safe_round(gg.loc[gg["avg_view_duration_sec"] > 0, "avg_view_duration_sec"].mean(), 1),
                "avg_engagement": _safe_round(gg["engagement_rate"].mean(), 3),
                "avg_retention_30s": _safe_round(ret["retention_30s"].mean(), 1) if len(ret) else None,
            })
        out[str(int(y))] = sorted(rows, key=lambda r: r["total_views"], reverse=True)
    return out


def compute_creator_2026(df: pd.DataFrame) -> dict:
    g = df[df["year"] == 2026].copy()
    if not len(g):
        return {
            "num_videos": 0, "total_views": 0, "avg_views": 0,
            "avg_avd": 0, "avg_engagement": 0, "total_subs": 0,
            "monthly": [], "genres": [], "top_videos": [],
        }
    monthly = []
    for m, mg in g.groupby("month"):
        monthly.append({
            "month": int(m),
            "videos": int(len(mg)),
            "views": _safe_int(mg["views"].sum()),
            "avg_views": _safe_round(mg["views"].mean(), 0),
        })
    monthly.sort(key=lambda r: r["month"])

    genres = []
    for genre, gg in g.groupby("game_genre"):
        genres.append({
            "game_genre": genre,
            "num": int(len(gg)),
            "avg_views": _safe_round(gg["views"].mean(), 0),
        })
    genres.sort(key=lambda r: r["game_genre"])

    top = g.nlargest(10, "views")
    top_videos = []
    for _, r in top.iterrows():
        top_videos.append({
            "title": str(r.get("title") or "")[:120],
            "game_name": r.get("game_name") or "",
            "game_genre": r.get("game_genre") or "",
            "views": _safe_int(r.get("views")),
            "likes": _safe_int(r.get("likes")),
            "engagement_rate": float(r.get("engagement_rate") or 0),
            "avg_view_duration_sec": _safe_int(r.get("avg_view_duration_sec")),
            "published_at": r["_pub"].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(r["_pub"]) else "",
        })

    return {
        "num_videos": int(len(g)),
        "total_views": _safe_int(g["views"].sum()),
        "avg_views": _safe_round(g["views"].mean(), 0),
        "avg_avd": _safe_round(g.loc[g["avg_view_duration_sec"] > 0, "avg_view_duration_sec"].mean(), 1),
        "avg_engagement": _safe_round(g["engagement_rate"].mean(), 3),
        "total_subs": _safe_int(g["subscribers_gained"].sum()),
        "monthly": monthly,
        "genres": genres,
        "top_videos": top_videos,
    }


def compute_performance_2026(df: pd.DataFrame) -> dict:
    g = df[df["year"] == 2026].copy()
    if not len(g):
        return {
            "avg_views": 0, "avg_avd_pct": 0, "total_videos": 0,
            "videos": [], "by_genre": [],
        }
    mean_views = float(g["views"].mean()) if len(g) else 0
    videos = []
    for _, r in g.sort_values("_pub").iterrows():
        v = _safe_int(r.get("views"))
        vs_avg = ((v - mean_views) / mean_views * 100) if mean_views else 0
        videos.append({
            "title": str(r.get("title") or "")[:64],  # match existing 64-char trim
            "game": r.get("game_name") or "",
            "genre": r.get("game_genre") or "",
            "date": r["_pub"].strftime("%Y-%m-%d") if pd.notna(r["_pub"]) else "",
            "views": v,
            "vs_avg": _safe_round(vs_avg, 1),
            "avd": _safe_int(r.get("avg_view_duration_sec")),
            "avd_pct": _safe_round(r.get("avg_view_percentage"), 1),
            "engagement": _safe_round(r.get("engagement_rate"), 2),
            "retention_30s": (
                _safe_round(r.get("retention_30s"), 1)
                if (pd.notna(r.get("retention_30s")) and float(r.get("retention_30s") or 0) > 0)
                else None
            ),
        })
    by_genre = []
    for genre, gg in g.groupby("game_genre"):
        by_genre.append({
            "game_genre": genre,
            "count": int(len(gg)),
            "avg_views": _safe_round(gg["views"].mean(), 1),
            "total_views": _safe_int(gg["views"].sum()),
        })
    by_genre.sort(key=lambda r: r["count"], reverse=True)

    return {
        "avg_views": _safe_round(mean_views, 0),
        "avg_avd_pct": _safe_round(g["avg_view_percentage"].mean(), 1),
        "total_videos": int(len(g)),
        "videos": videos,
        "by_genre": by_genre,
    }


def compute_top100(df: pd.DataFrame) -> list:
    top = df.nlargest(100, "views")
    out = []
    for _, r in top.iterrows():
        out.append({
            "title": str(r.get("title") or ""),
            "video_id": r.get("video_id") or "",
            "year": int(r["year"]) if pd.notna(r.get("year")) else None,
            "published_at": r["_pub"].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(r["_pub"]) else "",
            "views": _safe_int(r.get("views")),
            "likes": _safe_int(r.get("likes")),
            "comments": _safe_int(r.get("comments")),
            "subscribers_gained": _safe_int(r.get("subscribers_gained")),
            "engagement_rate": float(r.get("engagement_rate") or 0),
            "avg_view_duration_sec": _safe_int(r.get("avg_view_duration_sec")),
            "avg_view_percentage": _safe_round(r.get("avg_view_percentage"), 2),
            "retention_30s": _safe_round(r.get("retention_30s"), 2) if pd.notna(r.get("retention_30s")) else None,
            "retention_1min": _safe_round(r.get("retention_1min"), 2) if pd.notna(r.get("retention_1min")) else None,
            "retention_50pct": _safe_round(r.get("retention_50pct"), 2) if pd.notna(r.get("retention_50pct")) else None,
            "retention_70pct": _safe_round(r.get("retention_70pct"), 2) if pd.notna(r.get("retention_70pct")) else None,
            "game_name": r.get("game_name") or "",
            "game_genre": r.get("game_genre") or "",
            "video_format_label": r.get("video_format_label") or "",
            "visual_style_label": r.get("visual_style_label") or "",
        })
    return out


def compute_quarterly(df: pd.DataFrame) -> list:
    rows = []
    for (y, q), grp in df.groupby(["year", "quarter"]):
        if y < 2021:
            continue
        start, end = quarter_dates(int(y), int(q))
        days_total = (end - start).days + 1
        if is_current_quarter(int(y), int(q), TODAY):
            days_elapsed = (TODAY - start).days + 1
            factor = days_total / max(days_elapsed, 1)
            projected = True
        else:
            days_elapsed = days_total
            factor = 1.0
            projected = False

        ret = grp[grp["retention_30s"] > 0]
        ret_avg = float(ret["retention_30s"].mean()) if len(ret) else None
        retention_is_proxy = bool(len(ret) < max(3, len(grp) // 3))
        # Retention 1min (60s)
        ret1m_data = grp[grp["retention_1min"] > 0] if "retention_1min" in grp.columns else grp.iloc[0:0]
        ret_1min_avg = float(ret1m_data["retention_1min"].mean()) if len(ret1m_data) else None
        # Leave None when no data — chart will show a gap rather than a misleading flat 50%

        top = grp.nlargest(1, "views")
        top_title = str(top.iloc[0].get("title") or "")[:80] if len(top) else ""
        top_views = _safe_int(top.iloc[0].get("views")) if len(top) else 0

        num_videos = int(len(grp))
        total_views = _safe_int(grp["views"].sum())

        row = {
            "year": int(y),
            "quarter": int(q),
            "label": f"Q{int(q)} {int(y)}",
            "num_videos": num_videos,
            "total_views": total_views,
            "avg_views": _safe_round(grp["views"].mean(), 0),
            "avg_engagement": _safe_round(grp["engagement_rate"].mean(), 3),
            "avg_avd": _safe_round(grp.loc[grp["avg_view_duration_sec"] > 0, "avg_view_duration_sec"].mean(), 1),
            "avg_avd_sec": _safe_round(grp.loc[grp["avg_view_duration_sec"] > 0, "avg_view_duration_sec"].mean(), 1),
            "avg_retention_30s": (round(float(ret_avg), 1) if ret_avg is not None else None),
            "avg_retention_1min": (round(float(ret_1min_avg), 1) if ret_1min_avg is not None else None),
            "retention_is_proxy": retention_is_proxy,
            "top_video_title": top_title,
            "top_video_views": top_views,
            "projected": projected,
            "projected_factor": round(factor, 2),
            "days_elapsed": int(days_elapsed),
            "days_total": int(days_total),
            "num_videos_projected": int(round(num_videos * factor)),
            "total_views_projected": int(round(total_views * factor)),
        }
        rows.append(row)
    return sorted(rows, key=lambda r: (r["year"], r["quarter"]))


def compute_format_performance(df: pd.DataFrame) -> list:
    rows = []
    for fmt, grp in df.groupby("video_format_label"):
        if not fmt or (isinstance(fmt, float) and math.isnan(fmt)):
            continue
        rows.append({
            "video_format_label": fmt,
            "num_videos": int(len(grp)),
            "total_views": _safe_int(grp["views"].sum()),
            "avg_views": _safe_round(grp["views"].mean(), 1),
            "avg_avd": _safe_round(grp.loc[grp["avg_view_duration_sec"] > 0, "avg_view_duration_sec"].mean(), 1),
            "avg_engagement": _safe_round(grp["engagement_rate"].mean(), 1),
        })
    return sorted(rows, key=lambda r: r["avg_views"], reverse=True)


def compute_cross(df: pd.DataFrame) -> list:
    """Top 30 (genre, format, visual_style) combos by avg_views (min 3 vids)."""
    rows = []
    grouped = df.groupby(["game_genre", "video_format_label", "visual_style_label"])
    for (genre, fmt, style), grp in grouped:
        if not genre or not fmt or not style:
            continue
        if len(grp) < 3:
            continue
        rows.append({
            "game_genre": genre,
            "video_format_label": fmt,
            "visual_style_label": style,
            "num_videos": int(len(grp)),
            "total_views": _safe_int(grp["views"].sum()),
            "avg_views": _safe_round(grp["views"].mean(), 1),
            "avg_avd": _safe_round(grp.loc[grp["avg_view_duration_sec"] > 0, "avg_view_duration_sec"].mean(), 1),
            "avg_engagement": _safe_round(grp["engagement_rate"].mean(), 1),
        })
    rows.sort(key=lambda r: r["avg_views"], reverse=True)
    return rows[:30]


def compute_funnel(df: pd.DataFrame) -> dict:
    """3-stage funnel: views → engaged (~50% retention) → subs gained.

    Note: original strategic_data.json had this as concrete totals. We can't
    perfectly reconstruct YouTube's internal "engaged" definition without the
    Analytics API impressions/engaged-views report, so we approximate engaged
    as sum(views * retention_50pct/100) where retention is known, falling back
    to views * 0.17 (the historical engaged/total ratio of 16.9%) for videos
    without retention data.
    """
    total_views = _safe_int(df["views"].sum())
    total_subs = _safe_int(df["subscribers_gained"].sum())
    ret = df[df["retention_50pct"] > 0]
    no_ret = df[~(df["retention_50pct"] > 0)]
    engaged = (ret["views"] * ret["retention_50pct"] / 100).sum() + no_ret["views"].sum() * 0.17
    return {
        "total_views": total_views,
        "total_engaged": _safe_int(engaged),
        "total_subs": total_subs,
    }


def compute_traffic_detailed(df: pd.DataFrame) -> list:
    """Per-year traffic source breakdown using parsed traffic_sources column.

    browse  = YT_BROWSE + NO_LINK_OTHER + YT_CHANNEL + YT_OTHER_PAGE
    suggest = YT_SUGGESTED + RELATED_VIDEO
    search  = YT_SEARCH
    """
    BROWSE = ("YT_BROWSE", "NO_LINK_OTHER", "YT_CHANNEL", "YT_OTHER_PAGE")
    SUGGEST = ("YT_SUGGESTED", "RELATED_VIDEO")
    SEARCH = ("YT_SEARCH",)
    rows = []
    for y, grp in df.groupby("year"):
        ts_dicts = grp["traffic_sources"].apply(parse_traffic) if "traffic_sources" in grp.columns else []
        browse_abs = sum(_ts_views(ts, k) for ts in ts_dicts for k in BROWSE)
        sug_abs = sum(_ts_views(ts, k) for ts in ts_dicts for k in SUGGEST)
        sea_abs = sum(_ts_views(ts, k) for ts in ts_dicts for k in SEARCH)
        denom = max(_safe_int(grp["views"].sum()), 1)
        rows.append({
            "year": int(y),
            "n": int(len(grp)),
            "browse_abs": int(browse_abs),
            "browse_pct": _safe_round(browse_abs / denom * 100, 1),
            "suggested_abs": int(sug_abs),
            "suggested_pct": _safe_round(sug_abs / denom * 100, 1),
            "search_abs": int(sea_abs),
            "search_pct": _safe_round(sea_abs / denom * 100, 1),
        })
    return sorted(rows, key=lambda r: r["year"])


def compute_traffic_sources(df: pd.DataFrame) -> dict:
    """Top-level traffic_sources block: overall, by_genre, yearly (= traffic_detailed),
    plus benchmarks. Yearly mirrors traffic_detailed for backwards-compat with
    older dashboard code that reads either."""
    BROWSE = ("YT_BROWSE", "NO_LINK_OTHER", "YT_CHANNEL", "YT_OTHER_PAGE")
    SUGGEST = ("YT_SUGGESTED", "RELATED_VIDEO")
    SEARCH = ("YT_SEARCH",)

    ts_all = df["traffic_sources"].apply(parse_traffic) if "traffic_sources" in df.columns else []
    total_views = max(_safe_int(df["views"].sum()), 1)
    overall = {
        "browse_pct": _safe_round(sum(_ts_views(ts, k) for ts in ts_all for k in BROWSE) / total_views * 100, 1),
        "suggested_pct": _safe_round(sum(_ts_views(ts, k) for ts in ts_all for k in SUGGEST) / total_views * 100, 1),
        "search_pct": _safe_round(sum(_ts_views(ts, k) for ts in ts_all for k in SEARCH) / total_views * 100, 1),
    }

    by_genre = []
    for g, grp in df.groupby("game_genre"):
        ts_g = grp["traffic_sources"].apply(parse_traffic)
        denom = max(_safe_int(grp["views"].sum()), 1)
        by_genre.append({
            "genre": g,
            "num_videos": int(len(grp)),
            "browse_pct": _safe_round(sum(_ts_views(ts, k) for ts in ts_g for k in BROWSE) / denom * 100, 1),
            "suggested_pct": _safe_round(sum(_ts_views(ts, k) for ts in ts_g for k in SUGGEST) / denom * 100, 1),
            "search_pct": _safe_round(sum(_ts_views(ts, k) for ts in ts_g for k in SEARCH) / denom * 100, 1),
        })
    by_genre.sort(key=lambda r: r["browse_pct"], reverse=True)

    return {
        "overall": overall,
        "by_genre": by_genre,
        "yearly": compute_traffic_detailed(df),
        "benchmarks": {"browse_good": 40, "search_good": 15, "suggested_good": 20},
    }


# --------------------------------------------------------------------------
# advanced_metrics
# --------------------------------------------------------------------------

def compute_advanced_metrics(df: pd.DataFrame) -> dict:
    out = {}

    # avd_relative.yearly: AVD vs duration ratio
    avd_rel = []
    for y, grp in df.groupby("year"):
        dur = grp.loc[grp["duration_seconds"] > 0, "duration_seconds"]
        avd = grp.loc[grp["avg_view_duration_sec"] > 0, "avg_view_duration_sec"]
        avd_mean = float(avd.mean()) if len(avd) else 0
        dur_mean = float(dur.mean()) if len(dur) else 0
        pct = (avd_mean / dur_mean * 100) if dur_mean else 0
        avd_rel.append({
            "year": int(y),
            "avd_sec": _safe_round(avd_mean, 1),
            "dur_sec": _safe_round(dur_mean, 1),
            "pct": _safe_round(pct, 1),
        })
    avd_rel.sort(key=lambda r: r["year"])
    out["avd_relative"] = {"yearly": avd_rel}

    # consistency: weeks-with-publication / total weeks per year
    consistency = []
    for y, grp in df.groupby("year"):
        # weeks elapsed in that year
        if y == CURRENT_YEAR:
            total_weeks = max(int(TODAY.isocalendar()[1]), 1)
        else:
            total_weeks = 52
        weeks_with = grp["week"].nunique()
        consistency.append({
            "year": int(y),
            "total_weeks": int(total_weeks),
            "weeks_with": int(weeks_with),
            "weeks_without": int(total_weeks - weeks_with),
            "pct": _safe_round(weeks_with / total_weeks * 100, 1),
        })
    consistency.sort(key=lambda r: r["year"])
    out["consistency"] = consistency

    # efficiency: views/video, eng, subs/1k, min/video per year + trend arrows
    eff_raw = []
    for y, grp in df.groupby("year"):
        views_pv = float(grp["views"].mean()) if len(grp) else 0
        eng = float(grp["engagement_rate"].mean()) if len(grp) else 0
        total_views = float(grp["views"].sum())
        total_subs = float(grp["subscribers_gained"].sum())
        min_pv = float(grp["estimated_minutes_watched"].mean()) if len(grp) else 0
        eff_raw.append({
            "year": int(y),
            "num_videos": int(len(grp)),
            "views_per_video": _safe_round(views_pv, 0),
            "engagement": _safe_round(eng, 2),
            "subs_per_1k": _safe_round(total_subs / max(total_views, 1) * 1000, 2),
            "min_per_video": _safe_round(min_pv, 0),
        })
    eff_raw.sort(key=lambda r: r["year"])
    # Trend arrows vs previous year
    for i, r in enumerate(eff_raw):
        if i == 0:
            continue
        prev = eff_raw[i - 1]
        r["vpv_trend"] = "up" if r["views_per_video"] > prev["views_per_video"] else "down"
        r["eng_trend"] = "up" if r["engagement"] > prev["engagement"] else "down"
        r["spk_trend"] = "up" if r["subs_per_1k"] > prev["subs_per_1k"] else "down"
    out["efficiency"] = eff_raw

    # health: per-month score 0-100 from avd, eng, retention, frequency
    # Match the existing schema: {avd, eng, freq, ret, score, label, month, year, n}
    health = []
    df_y_min = int(df["year"].min())
    avd_p90 = df.loc[df["avg_view_duration_sec"] > 0, "avg_view_duration_sec"].quantile(0.9) or 1
    eng_p90 = df["engagement_rate"].quantile(0.9) or 1
    for (y, m), grp in df.groupby(["year", "month"]):
        if y < df_y_min:
            continue
        avd_m = grp.loc[grp["avg_view_duration_sec"] > 0, "avg_view_duration_sec"].mean()
        eng_m = grp["engagement_rate"].mean()
        ret_m = grp.loc[grp["retention_30s"] > 0, "retention_30s"].mean()
        n = int(len(grp))
        # frequency: % of weeks in month with at least one publish (max 4)
        weeks_m = grp["week"].nunique()
        # weeks in this month: ~4 (varies); use 4 baseline, capped at 100
        freq_pct = min(weeks_m / 4 * 100, 100.0)
        avd_score = min((avd_m or 0) / float(avd_p90) * 100, 100)
        eng_score = min((eng_m or 0) / float(eng_p90) * 100, 100)
        ret_score = min((ret_m or 50) / 65 * 100, 100)  # 65% retention → 100
        score = round(avd_score * 0.30 + eng_score * 0.30 + freq_pct * 0.25 + ret_score * 0.15, 1)
        health.append({
            "year": int(y),
            "month": int(m),
            "label": f"{MONTH_ABBR_ES[int(m) - 1]} {int(y)}",
            "n": n,
            "avd": int(round(avd_score)) if not math.isnan(avd_score) else 0,
            "eng": int(round(eng_score)) if not math.isnan(eng_score) else 0,
            "freq": int(round(freq_pct)),
            "ret": int(round(ret_score)) if not math.isnan(ret_score) else 0,
            "score": score if not math.isnan(score) else 0,
        })
    health.sort(key=lambda r: (r["year"], r["month"]))
    out["health"] = health

    # outliers: viral (>2σ) vs underperforming (<0.3×mean)
    mean_v = float(df["views"].mean())
    std_v = float(df["views"].std())
    threshold_viral = mean_v + 2 * std_v
    threshold_under = mean_v * 0.3
    viral = df[df["views"] >= threshold_viral].sort_values("views", ascending=False)
    under = df[df["views"] < threshold_under]
    viral_videos = []
    for _, r in viral.head(15).iterrows():
        viral_videos.append({
            "title": str(r.get("title") or "")[:64],
            "views": _safe_int(r.get("views")),
            "year": int(r["year"]) if pd.notna(r.get("year")) else None,
            "game": r.get("game_name") or "",
            "genre": r.get("game_genre") or "",
            "format": r.get("video_format_label") or "",
        })
    out["outliers"] = {
        "mean": _safe_round(mean_v, 0),
        "std": _safe_round(std_v, 0),
        "threshold_viral": int(round(threshold_viral)),
        "threshold_under": int(round(threshold_under)),
        "viral_count": int(len(viral)),
        "under_count": int(len(under)),
        "viral_genres": dict(Counter(viral["game_genre"].dropna())),
        "viral_formats": dict(Counter(viral["video_format_label"].dropna())),
        "viral_videos": viral_videos,
    }

    # seasonal_genre: avg_views per month per genre
    seasonal = {}
    for genre, grp in df.groupby("game_genre"):
        rows = []
        for m in range(1, 13):
            mg = grp[grp["month"] == m]
            rows.append({
                "month": m,
                "n": int(len(mg)),
                "avg_views": _safe_round(mg["views"].mean(), 0) if len(mg) else 0,
            })
        seasonal[genre] = rows
    out["seasonal_genre"] = seasonal

    # sub_conversion: subs_per_1k_views by genre
    sub_conv = []
    for g, grp in df.groupby("game_genre"):
        tv = float(grp["views"].sum())
        ts = float(grp["subscribers_gained"].sum())
        sub_conv.append({
            "genre": g,
            "n": int(len(grp)),
            "total_views": int(tv),
            "total_subs": int(ts),
            "subs_per_1k": _safe_round(ts / max(tv, 1) * 1000, 2),
        })
    sub_conv.sort(key=lambda r: r["subs_per_1k"], reverse=True)
    out["sub_conversion"] = sub_conv

    # winners: % of videos above year's avg_views
    winners = []
    for y, grp in df.groupby("year"):
        avg = float(grp["views"].mean()) if len(grp) else 0
        above = int((grp["views"] > avg).sum())
        winners.append({
            "year": int(y),
            "total": int(len(grp)),
            "avg": _safe_round(avg, 0),
            "above": above,
            "pct": _safe_round(above / max(len(grp), 1) * 100, 1),
        })
    winners.sort(key=lambda r: r["year"])
    out["winners"] = winners

    # traffic_detailed (also exposed at top-level — keep here for compat)
    out["traffic_detailed"] = compute_traffic_detailed(df)

    # interpretations: derived flags pointing to specific entries
    cur_eff = next((e for e in eff_raw if e["year"] == CURRENT_YEAR), eff_raw[-1] if eff_raw else None)
    best_conv = max(eff_raw, key=lambda e: e["subs_per_1k"]) if eff_raw else None
    best_sub = sub_conv[0] if sub_conv else None
    cur_health = next((h for h in reversed(health) if h["year"] == CURRENT_YEAR), health[-1] if health else None)
    cons_24 = next((c for c in consistency if c["year"] == 2024), None)
    cons_cur = next((c for c in consistency if c["year"] == CURRENT_YEAR), None)
    seas_peaks = {}
    for g, rows in seasonal.items():
        peak = max(rows, key=lambda r: r["avg_views"])
        seas_peaks[g] = {
            "month_num": peak["month"],
            "month": MONTH_ABBR_ES[peak["month"] - 1],
            "views": peak["avg_views"],
        }
    score_cur = (cur_health or {}).get("score", 0) or 0
    health_status = "green" if score_cur >= 80 else ("yellow" if score_cur >= 60 else "red")
    out["interpretations"] = {
        "current_conversion": cur_eff,
        "best_conversion_year": best_conv,
        "best_sub_genre": best_sub,
        "consistency_2024": cons_24,
        f"consistency_{CURRENT_YEAR}": cons_cur,
        "health_current": cur_health,
        "health_status": health_status,
        "seasonal_peaks": seas_peaks,
    }

    return out


# --------------------------------------------------------------------------
# channel_activity*
# --------------------------------------------------------------------------

def compute_channel_activity_quarterly(df: pd.DataFrame) -> list:
    rows = []
    for (y, q), grp in df.groupby(["year", "quarter"]):
        if y < 2021:
            continue
        start, end = quarter_dates(int(y), int(q))
        total = (end - start).days + 1
        if is_current_quarter(int(y), int(q), TODAY):
            elapsed = max(1, (TODAY - start).days + 1)
            factor = total / elapsed
            is_current = True
        else:
            elapsed = total
            factor = 1.0
            is_current = False
        views = _safe_int(grp["views"].sum())
        minutes = _safe_int(grp["estimated_minutes_watched"].sum())
        subs = _safe_int(grp["subscribers_gained"].sum())
        rows.append({
            "year": int(y),
            "quarter": int(q),
            "label": f"Q{int(q)} {int(y)}",
            "views": views,
            "minutes": minutes,
            "subs_gained": subs,
            "views_projected": int(round(views * factor)),
            "minutes_projected": int(round(minutes * factor)),
            "subs_gained_projected": int(round(subs * factor)),
            "projected": is_current,
            "projected_factor": round(factor, 2),
            "days_elapsed": elapsed,
            "days_total": total,
        })
    return sorted(rows, key=lambda r: (r["year"], r["quarter"]))


def compute_channel_activity_quarterly_3y(df: pd.DataFrame) -> list:
    cutoff = CURRENT_YEAR - 2
    return [r for r in compute_channel_activity_quarterly(df) if r["year"] >= cutoff]


def update_channel_activity(df: pd.DataFrame, existing: dict | None) -> dict:
    """Add/update only the LATEST month rows from CSV, keeping historical
    Analytics-API rows (likes/minutes_watched/subs_gained/subs_lost) untouched.

    The CSV doesn't have channel-level monthly Analytics data (subs_lost is
    only available via the YouTube Analytics API), so we DON'T touch the
    historical channel_activity.monthly / yearly figures unless the row is
    for the current year, in which case we refresh the per-month aggregate
    using CSV data (likes from videos, minutes from estimated_minutes_watched,
    subs_gained from subscribers_gained, subs_lost preserved if previously
    tracked or set to 0).

    To stay safe, this function returns existing untouched if existing is
    well-formed and includes a row for the current month — historical data
    came from the YouTube Analytics API and we don't want to overwrite it
    with potentially less-accurate per-video sums.
    """
    if existing and isinstance(existing, dict) and existing.get("monthly") and existing.get("yearly"):
        # leave intact — historical Analytics API data is more authoritative.
        return existing
    # Fallback: build from CSV (less accurate, used only when missing)
    monthly = []
    for (y, m), grp in df.groupby(["year", "month"]):
        if y < 2021:
            continue
        monthly.append({
            "year": int(y),
            "month": int(m),
            "views": _safe_int(grp["views"].sum()),
            "likes": _safe_int(grp["likes"].sum()),
            "minutes_watched": _safe_int(grp["estimated_minutes_watched"].sum()),
            "subs_gained": _safe_int(grp["subscribers_gained"].sum()),
            "subs_lost": 0,
        })
    monthly.sort(key=lambda r: (r["year"], r["month"]))
    yearly = []
    for y, grp in df.groupby("year"):
        if y < 2021:
            continue
        sg = _safe_int(grp["subscribers_gained"].sum())
        yearly.append({
            "year": int(y),
            "total_views": _safe_int(grp["views"].sum()),
            "total_likes": _safe_int(grp["likes"].sum()),
            "total_minutes": _safe_int(grp["estimated_minutes_watched"].sum()),
            "subs_gained": sg,
            "subs_lost": 0,
            "net_subs": sg,
        })
    yearly.sort(key=lambda r: r["year"])
    return {"monthly": monthly, "yearly": yearly}


# --------------------------------------------------------------------------
# Audience heatmap (manual YouTube Studio data)
# --------------------------------------------------------------------------

DAYS_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_ES = {"Mon": "Lun", "Tue": "Mar", "Wed": "Mié", "Thu": "Jue",
          "Fri": "Vie", "Sat": "Sáb", "Sun": "Dom"}
HOUR_BUCKETS = [0, 3, 6, 9, 12, 15, 18, 21]
LEAD_TIME_HOURS = 3  # YouTube needs ~3hr to index a new upload before recommendations kick in


def _interp_intensity(intensity_by_day: dict, day: str, hour: int) -> float:
    """Linear interpolation between adjacent 3-hour buckets for a given day,
    wrapping around midnight (e.g., hour 22 interpolates between 21 and 0+24).
    """
    day_data = intensity_by_day.get(day, {})
    # Convert string keys to ints for lookups
    int_data = {int(k): float(v) for k, v in day_data.items()}
    if hour in int_data:
        return float(int_data[hour])
    # Find bracket
    lower = max((b for b in HOUR_BUCKETS if b <= hour), default=21)
    upper = min((b for b in HOUR_BUCKETS if b > hour), default=None)
    if upper is None:
        # interpolate between 21 (today) and 0 (next day, treated as +24)
        upper_val = int_data.get(0, 0)
        lower_val = int_data.get(21, 0)
        span = 24 - 21
        frac = (hour - 21) / span
        return lower_val + (upper_val - lower_val) * frac
    span = upper - lower
    frac = (hour - lower) / span if span else 0
    return int_data.get(lower, 0) + (int_data.get(upper, 0) - int_data.get(lower, 0)) * frac


def compute_audience_recommendations(df: pd.DataFrame, manual_path: Path) -> dict | None:
    """Load the manual YouTube Studio heatmap and derive top-3 publish windows.

    Returns dict with keys: audience_heatmap, publish_recommendations, manual_data_meta.
    Returns None if manual file is missing or invalid (caller logs warning).
    """
    if not manual_path.exists():
        print(f"  WARN: {manual_path.name} not found — skipping audience recommendations")
        return None
    try:
        with open(manual_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        meta = raw.get("_meta") or {}
        intensity = raw.get("intensity") or {}
        # Validate: 7 days × 8 buckets
        for d in DAYS_ORDER:
            if d not in intensity:
                raise ValueError(f"missing day '{d}'")
            for hb in HOUR_BUCKETS:
                if str(hb) not in intensity[d]:
                    raise ValueError(f"missing hour {hb} for day {d}")
                v = intensity[d][str(hb)]
                if not isinstance(v, (int, float)) or v < 0 or v > 100:
                    raise ValueError(f"intensity out of range for {d} {hb}: {v}")
    except Exception as exc:
        print(f"  WARN: failed to parse {manual_path.name}: {exc} — skipping")
        return None

    # Build the grid (7 rows × 8 cols)
    grid = [[int(intensity[d][str(h)]) for h in HOUR_BUCKETS] for d in DAYS_ORDER]
    audience_heatmap = {
        "_meta": meta,
        "days": list(DAYS_ORDER),
        "hours": list(HOUR_BUCKETS),
        "grid": grid,
    }

    # Score each (day, publish_hour) pair: audience intensity at publish_hour + LEAD_TIME_HOURS
    # Iterate every 24 hours for richer scoring; interpolate if hour not on grid.
    scored = []
    for di, day in enumerate(DAYS_ORDER):
        for pub_h in range(24):
            tgt_h = pub_h + LEAD_TIME_HOURS
            tgt_day = day
            if tgt_h >= 24:
                tgt_h -= 24
                tgt_day = DAYS_ORDER[(di + 1) % 7]
            score = _interp_intensity(intensity, tgt_day, tgt_h)
            scored.append({
                "day": day,
                "publish_hour": pub_h,
                "audience_day": tgt_day,
                "audience_hour": tgt_h,
                "score": score,
            })

    # Pick top 3, but enforce (day, ~publish_hour bucket) diversity so we don't
    # return three near-identical adjacent slots — group by day first.
    scored.sort(key=lambda r: r["score"], reverse=True)
    top = []
    seen_days = set()
    for s in scored:
        # First pass: prefer one slot per day
        if s["day"] in seen_days:
            continue
        top.append(s)
        seen_days.add(s["day"])
        if len(top) >= 3:
            break
    # If fewer than 3 (very small grid), backfill from sorted
    if len(top) < 3:
        for s in scored:
            if s not in top:
                top.append(s)
                if len(top) >= 3:
                    break

    recs = []
    for r in top:
        score = float(r["score"])
        if score >= 80:
            conf = "alta"
        elif score >= 60:
            conf = "media"
        else:
            conf = "baja"
        pub_h = r["publish_hour"]
        aud_h = r["audience_hour"]
        aud_day_es = DAY_ES.get(r["audience_day"], r["audience_day"])
        same_day = r["audience_day"] == r["day"]
        # Audience window = 3hr block starting at the audience hour
        aud_end = (aud_h + 3) % 24
        window_label = f"{aud_h:02d}:00-{aud_end:02d}:00 Chile"
        if not same_day:
            window_label = f"{aud_day_es} {window_label}"
        reasoning = (
            f"Publicar {DAY_ES[r['day']]} {pub_h:02d}:00 hace que el video aparezca en feeds "
            f"alrededor de las {aud_h:02d}:00 (lead time +{LEAD_TIME_HOURS}hr de YouTube), "
            f"justo cuando tu audiencia tiene intensidad {int(round(score))}/100 según YouTube Studio."
        )
        recs.append({
            "day": r["day"],
            "day_es": DAY_ES.get(r["day"], r["day"]),
            "hour": f"{pub_h:02d}:00",
            "publish_hour_int": pub_h,
            "audience_window": window_label,
            "audience_hour_int": aud_h,
            "score": round(score, 1),
            "confidence": conf,
            "reasoning": reasoning,
        })

    manual_data_meta = {
        "last_updated": meta.get("last_updated", "--"),
        "source": meta.get("source", "YouTube Studio"),
        "timezone": meta.get("timezone", "America/Santiago (GMT-4)"),
        "period": meta.get("period", "Últimos 28 días"),
    }

    return {
        "audience_heatmap": audience_heatmap,
        "publish_recommendations": recs,
        "manual_data_meta": manual_data_meta,
    }


def compute_recent_videos_retention(df: pd.DataFrame, weeks: int = 4) -> dict:
    """Per-video AVD + retention metrics for the last N weeks of publications.

    Always returns at least the most recent videos so the dashboard always
    shows the latest content even if it falls outside the strict window.
    Missing values stay as None — the UI renders 'no existe información'.
    """
    cutoff = TODAY - timedelta(days=weeks * 7)
    recent = df[df["date"] >= cutoff].copy()
    # Safety net: if 4-week window is empty, fall back to the last 5 published
    if len(recent) == 0:
        recent = df.sort_values("_pub", ascending=False).head(5).copy()
    recent = recent.sort_values("_pub", ascending=False)

    videos = []
    for _, r in recent.iterrows():
        videos.append({
            "video_id": str(r.get("video_id") or ""),
            "title": str(r.get("title") or "")[:100],
            "date": r["_pub"].strftime("%Y-%m-%d"),
            "views": _safe_int(r.get("views")),
            "duration_seconds": _safe_int(r.get("duration_seconds")),
            "avg_avd_sec": (round(float(r["avg_view_duration_sec"]), 1)
                            if pd.notna(r.get("avg_view_duration_sec")) and r.get("avg_view_duration_sec", 0) > 0
                            else None),
            "retention_30s": (round(float(r["retention_30s"]), 1)
                              if pd.notna(r.get("retention_30s")) and r.get("retention_30s", 0) > 0
                              else None),
            "retention_1min": (round(float(r["retention_1min"]), 1)
                               if pd.notna(r.get("retention_1min")) and r.get("retention_1min", 0) > 0
                               else None),
            "game_genre": str(r.get("game_genre") or ""),
        })

    return {
        "window_days": weeks * 7,
        "cutoff_date": cutoff.isoformat(),
        "today": TODAY.isoformat(),
        "videos": videos,
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    print(f"[recompute_aggregates] today = {TODAY.isoformat()}")
    df = load_data()
    print(f"  Loaded {len(df)} videos from {CSV.name}")
    print(f"  Year range: {int(df['year'].min())}-{int(df['year'].max())}")

    with open(JSON_PATH) as f:
        d = json.load(f)
    before_size = len(json.dumps(d, ensure_ascii=False))

    # Recompute
    d["overview"] = compute_overview(df)
    d["yearly_kpis"] = compute_yearly_kpis(df)
    d["yearly"] = compute_yearly(df)
    d["monthly_by_year"] = compute_monthly_by_year(df)
    d["genre_by_year"] = compute_genre_by_year(df)
    d["creator_2026"] = compute_creator_2026(df)
    d["performance_2026"] = compute_performance_2026(df)
    d["recent_videos_retention"] = compute_recent_videos_retention(df, weeks=4)
    d["top100"] = compute_top100(df)
    d["quarterly"] = compute_quarterly(df)
    d["format_performance"] = compute_format_performance(df)
    d["cross"] = compute_cross(df)
    d["funnel"] = compute_funnel(df)
    d["traffic_detailed"] = compute_traffic_detailed(df)
    d["traffic_sources"] = compute_traffic_sources(df)
    d["advanced_metrics"] = compute_advanced_metrics(df)
    d["channel_activity_quarterly"] = compute_channel_activity_quarterly(df)
    d["channel_activity_quarterly_3y"] = compute_channel_activity_quarterly_3y(df)
    d["channel_activity"] = update_channel_activity(df, d.get("channel_activity"))

    # Audience heatmap (manual YouTube Studio data) → publish recommendations
    aud = compute_audience_recommendations(df, MANUAL_HEATMAP_PATH)
    if aud is not None:
        d["audience_heatmap"] = aud["audience_heatmap"]
        d["publish_recommendations"] = aud["publish_recommendations"]
        d["manual_data_meta"] = aud["manual_data_meta"]
        print(f"  audience_heatmap injected ({len(aud['audience_heatmap']['days'])}×{len(aud['audience_heatmap']['hours'])})")
        print(f"  publish_recommendations: {len(aud['publish_recommendations'])} top windows")
        for i, r in enumerate(aud["publish_recommendations"], 1):
            print(f"    #{i} {r['day_es']} {r['hour']} → audiencia {r['audience_window']} (score {r['score']}, {r['confidence']})")

    # Audience demographics (manual YouTube Studio data)
    if MANUAL_DEMOGRAPHICS_PATH.exists():
        try:
            with open(MANUAL_DEMOGRAPHICS_PATH) as f:
                demo = json.load(f)
            d["audience_demographics"] = demo
            primary = demo.get("_derived", {}).get("primary_segment", "n/a")
            print(f"  audience_demographics injected (primary: {primary})")
        except Exception as e:
            print(f"  [warn] no pude leer demographics: {e}")

    # Write
    out_text = json.dumps(d, ensure_ascii=False)
    with open(JSON_PATH, "w") as f:
        f.write(out_text)
    after_size = len(out_text)

    # Diagnostics
    p26 = d["performance_2026"]
    last_date = max((v["date"] for v in p26["videos"]), default="(none)")
    proj_q = [q for q in d["quarterly"] if q.get("projected")]
    top100 = d["top100"]
    years_in_top = sorted({v["published_at"][:4] for v in top100 if v.get("published_at")})

    df_2026 = int((df["year"] == 2026).sum())
    print(f"\nstrategic_data.json updated: {after_size:,} bytes  (was {before_size:,}, Δ {after_size - before_size:+,})")
    print(f"  performance_2026: {len(p26['videos'])} videos through {last_date}  (CSV df[year==2026]={df_2026})")
    print(f"  quarterly: {len(d['quarterly'])} rows; projected current quarter:")
    for q in proj_q:
        print(f"    Q{q['quarter']} {q['year']}: days {q['days_elapsed']}/{q['days_total']}, "
              f"factor={q['projected_factor']}×, "
              f"videos {q['num_videos']}→{q['num_videos_projected']}, "
              f"views {q['total_views']:,}→{q['total_views_projected']:,}")
    print(f"  top100: {len(top100)} videos | year range {years_in_top[0] if years_in_top else '?'}-"
          f"{years_in_top[-1] if years_in_top else '?'}")
    if len(p26["videos"]) != df_2026:
        print(f"  WARNING: performance_2026 video count != df[year==2026] count ({len(p26['videos'])} vs {df_2026})")
    else:
        print(f"  OK: performance_2026 count matches CSV exactly")


if __name__ == "__main__":
    main()
