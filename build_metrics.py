#!/usr/bin/env python3
"""Regenerate metrics.html from videos_categorized.csv.

Computes aggregates that mirror the embedded `var D = {...}` JSON shape used by
the Chart.js charts in metrics.html, then templates the HTML around them.
The JS tail (var fmt = ... Chart() construction) is preserved verbatim from
the existing metrics.html so the charts keep working without changes.

Run from the repo root:
    python3 build_metrics.py
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "videos_categorized.csv")
STRATEGIC_JSON = os.path.join(BASE_DIR, "strategic_data.json")
TEMPLATE_HTML = os.path.join(BASE_DIR, "metrics.html")
OUTPUT_HTML = os.path.join(BASE_DIR, "metrics.html")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MONTHS_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
             "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

# Map YouTube Analytics traffic source codes to the three buckets the report
# tracks. We try YT_BROWSE / YT_SUGGESTED first because the spec mentions
# them, then fall back to the codes that actually appear in our CSV.
BROWSE_KEYS = ["YT_BROWSE", "NO_LINK_OTHER", "YT_CHANNEL", "YT_OTHER_PAGE"]
SUGGESTED_KEYS = ["YT_SUGGESTED", "RELATED_VIDEO"]
SEARCH_KEYS = ["YT_SEARCH"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def trend(curr: float | None, prev: float | None, eps: float = 0.01) -> str:
    """up / down / flat trend arrow vs prior year."""
    if curr is None or prev is None:
        return ""
    if abs(curr - prev) <= eps * max(abs(prev), 1):
        return "flat"
    return "up" if curr > prev else "down"


def trend_td(t: str) -> str:
    """Trend arrow as an HTML <td>."""
    if t == "up":
        return '<td class="pos">&#9650;</td>'
    if t == "down":
        return '<td class="neg">&#9660;</td>'
    if t == "flat":
        return '<td>&#9644;</td>'
    return '<td></td>'


def safe_int(x) -> int:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0
    return int(x)


def parse_traffic(raw) -> dict[str, float]:
    """Return {key: views} from a traffic_sources JSON cell."""
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        d = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    out: dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = float(v.get("views", 0) or 0)
        else:
            try:
                out[k] = float(v or 0)
            except (TypeError, ValueError):
                out[k] = 0.0
    return out


def sum_keys(traffic: dict[str, float], keys: list[str]) -> float:
    return sum(traffic.get(k, 0.0) for k in keys)


# ---------------------------------------------------------------------------
# 1. Load + prep
# ---------------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce").dt.tz_localize(None)
df = df.dropna(subset=["published_at"]).copy()
df["year"] = df["published_at"].dt.year.astype(int)
df["month"] = df["published_at"].dt.month.astype(int)
df["quarter"] = ((df["month"] - 1) // 3 + 1).astype(int)
df["yq"] = df["year"].astype(str) + "Q" + df["quarter"].astype(str)
iso = df["published_at"].dt.isocalendar()
df["iso_year"] = iso["year"].astype(int)
df["iso_week"] = iso["week"].astype(int)

# Load projection metadata from strategic_data.json (computed by recompute_aggregates.py)
try:
    with open(STRATEGIC_JSON, "r", encoding="utf-8") as _f:
        _strategic = json.load(_f)
    _quarterly_meta = {
        f"{q['year']}Q{q['quarter']}": q
        for q in _strategic.get("quarterly", [])
    }
except (FileNotFoundError, ValueError):
    _quarterly_meta = {}

# Numeric coercions
for col in ["views", "subscribers_gained", "estimated_minutes_watched",
            "avg_view_duration_sec", "duration_seconds", "engagement_rate",
            "retention_30s"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Pre-parse traffic_sources once
df["_traffic"] = df["traffic_sources"].apply(parse_traffic)
df["_browse_views"] = df["_traffic"].apply(lambda t: sum_keys(t, BROWSE_KEYS))
df["_suggested_views"] = df["_traffic"].apply(lambda t: sum_keys(t, SUGGESTED_KEYS))
df["_search_views"] = df["_traffic"].apply(lambda t: sum_keys(t, SEARCH_KEYS))

# Filter ALL aggregates to 2025+ (per dashboard simplification 2026-05)
df = df[df["year"] >= 2025].copy()
df["quarter"] = df["published_at"].dt.quarter

YEARS_ALL = sorted(df["year"].unique().tolist())
TODAY = datetime.now(timezone.utc).date()
CURRENT_YEAR = TODAY.year
CURRENT_QUARTER = (TODAY.month - 1) // 3 + 1
RECENT_YEARS = [y for y in [2025, 2026] if y in YEARS_ALL]


def _quarter_dates(y: int, q: int):
    from datetime import date, timedelta
    m_start = (q - 1) * 3 + 1
    start = date(y, m_start, 1)
    if m_start + 3 > 12:
        end = date(y, 12, 31)
    else:
        end = date(y, m_start + 3, 1) - timedelta(days=1)
    return start, end


def _quarter_projection(y: int, q: int):
    """Return (days_elapsed, days_total, factor, is_current) for a quarter."""
    start, end = _quarter_dates(y, q)
    total = (end - start).days + 1
    if y == CURRENT_YEAR and q == CURRENT_QUARTER:
        elapsed = max(1, (TODAY - start).days + 1)
        return elapsed, total, total / elapsed, True
    return total, total, 1.0, False

# ---------------------------------------------------------------------------
# 2. efficiency (per recent year)
# ---------------------------------------------------------------------------
efficiency: list[dict] = []
prev_eff: dict | None = None
for y in RECENT_YEARS:
    sub = df[df["year"] == y]
    n = len(sub)
    if n == 0:
        continue
    views_total = sub["views"].sum()
    subs_total = sub["subscribers_gained"].sum()
    vpv = round(sub["views"].mean(), 0) if n else 0
    spk = round((subs_total / views_total * 1000), 2) if views_total else 0
    eng = round(sub["engagement_rate"].mean(), 2) if n else 0
    mpv = round(sub["estimated_minutes_watched"].mean(), 0) if n else 0
    entry = {
        "year": int(y),
        "views_per_video": float(vpv),
        "subs_per_1k": float(spk),
        "engagement": float(eng),
        "min_per_video": float(mpv),
        "num_videos": int(n),
        "vpv_trend": trend(vpv, prev_eff["views_per_video"] if prev_eff else None),
        "spk_trend": trend(spk, prev_eff["subs_per_1k"] if prev_eff else None),
        "eng_trend": trend(eng, prev_eff["engagement"] if prev_eff else None),
    }
    efficiency.append(entry)
    prev_eff = entry

# ---------------------------------------------------------------------------
# 2b. quarterly efficiency (per (year, quarter), 2025+, with current-quarter projection)
# ---------------------------------------------------------------------------
quarterly_efficiency: list[dict] = []
prev_q_eff: dict | None = None
quarter_keys = sorted(df.dropna(subset=["quarter"]).groupby(["year", "quarter"]).groups.keys())
for (y, q) in quarter_keys:
    sub = df[(df["year"] == y) & (df["quarter"] == q)]
    n = len(sub)
    if n == 0:
        continue
    views_total = sub["views"].sum()
    subs_total = sub["subscribers_gained"].sum()
    vpv = round(sub["views"].mean(), 0) if n else 0
    spk = round((subs_total / views_total * 1000), 2) if views_total else 0
    eng = round(sub["engagement_rate"].mean(), 2) if n else 0
    elapsed, total, factor, is_current = _quarter_projection(int(y), int(q))
    entry = {
        "year": int(y),
        "quarter": int(q),
        "label": f"{int(y)}Q{int(q)}",
        "num_videos": int(n),
        "num_videos_projected": int(round(n * factor)),
        "total_views": int(views_total),
        "total_views_projected": int(round(views_total * factor)),
        "views_per_video": float(vpv),
        "subs_per_1k": float(spk),
        "engagement": float(eng),
        "vpv_trend": trend(vpv, prev_q_eff["views_per_video"] if prev_q_eff else None),
        "spk_trend": trend(spk, prev_q_eff["subs_per_1k"] if prev_q_eff else None),
        "eng_trend": trend(eng, prev_q_eff["engagement"] if prev_q_eff else None),
        "projected": is_current,
        "projected_factor": round(factor, 2),
        "days_elapsed": elapsed,
        "days_total": total,
    }
    quarterly_efficiency.append(entry)
    prev_q_eff = entry

# ---------------------------------------------------------------------------
# 3. health (per year-month from Jan 2024 forward)
# ---------------------------------------------------------------------------
health: list[dict] = []
hm = df[df["year"] >= 2025].groupby(["year", "month"])
for (y, m), sub in hm:
    n = len(sub)
    if n < 1:
        continue
    freq = min(100.0, n / 12 * 100)
    eng_avg = sub["engagement_rate"].mean()
    eng = min(100.0, (eng_avg / 4 * 100) if pd.notna(eng_avg) else 0.0)
    avd_avg = sub["avg_view_duration_sec"].mean()
    avd = min(100.0, (avd_avg / 90 * 100) if pd.notna(avd_avg) else 0.0)
    ret_series = sub["retention_30s"].dropna() if "retention_30s" in sub else pd.Series(dtype=float)
    ret_series = ret_series[ret_series > 0]
    if len(ret_series) > 0:
        ret = min(100.0, ret_series.mean() / 65 * 100)
    else:
        ret = 77.0
    score = round((freq + eng + avd + ret) / 4, 1)
    health.append({
        "year": int(y),
        "month": int(m),
        "label": f"{MONTHS_ES[m-1]} {y}",
        "score": float(score),
        "freq": round(freq, 1) if freq < 100 else 100,
        "eng": round(eng, 1) if eng < 100 else 100,
        "avd": round(avd, 1) if avd < 100 else 100,
        "ret": round(ret, 1) if ret < 100 else 100,
        "n": int(n),
    })

health.sort(key=lambda h: (h["year"], h["month"]))

# ---------------------------------------------------------------------------
# 4. winners
# ---------------------------------------------------------------------------
winners: list[dict] = []
for y in RECENT_YEARS:
    sub = df[df["year"] == y]
    if sub.empty:
        continue
    avg = sub["views"].mean()
    above = int((sub["views"] > avg).sum())
    total = int(len(sub))
    pct = round(above / total * 100, 1) if total else 0.0
    winners.append({
        "year": int(y),
        "pct": float(pct),
        "above": above,
        "total": total,
        "avg": round(float(avg), 0),
    })

# ---------------------------------------------------------------------------
# 5. sub_conversion (per genre, all data)
# ---------------------------------------------------------------------------
sub_conversion: list[dict] = []
for genre, sub in df.groupby("game_genre"):
    if pd.isna(genre):
        continue
    total_views = float(sub["views"].sum())
    total_subs = float(sub["subscribers_gained"].sum())
    n = int(len(sub))
    spk = round(total_subs / total_views * 1000, 2) if total_views > 0 else 0.0
    sub_conversion.append({
        "genre": str(genre),
        "subs_per_1k": float(spk),
        "total_subs": int(total_subs),
        "total_views": int(total_views),
        "n": n,
    })

sub_conversion.sort(key=lambda r: r["subs_per_1k"], reverse=True)

# ---------------------------------------------------------------------------
# 6. avd_relative.yearly
# ---------------------------------------------------------------------------
avd_yearly: list[dict] = []
for y in RECENT_YEARS:
    sub = df[df["year"] == y]
    if sub.empty:
        continue
    avd_sec = sub["avg_view_duration_sec"].mean()
    dur_sec = sub["duration_seconds"].mean()
    pct = round(avd_sec / dur_sec * 100, 1) if dur_sec else 0.0
    avd_yearly.append({
        "year": int(y),
        "pct": float(pct),
        "avd_sec": round(float(avd_sec), 1) if pd.notna(avd_sec) else 0.0,
        "dur_sec": round(float(dur_sec), 1) if pd.notna(dur_sec) else 0.0,
    })

avd_relative = {"yearly": avd_yearly}

# ---------------------------------------------------------------------------
# 7. consistency (per recent year)
# ---------------------------------------------------------------------------
consistency: list[dict] = []
for y in RECENT_YEARS:
    sub = df[df["year"] == y]
    if y < CURRENT_YEAR:
        total_weeks = 52
    else:
        total_weeks = int(TODAY.isocalendar()[1])
    weeks_with = int(sub["iso_week"].nunique())
    weeks_with = min(weeks_with, total_weeks)
    weeks_without = total_weeks - weeks_with
    pct = round(weeks_with / total_weeks * 100, 1) if total_weeks else 0.0
    consistency.append({
        "year": int(y),
        "total_weeks": total_weeks,
        "weeks_with": weeks_with,
        "weeks_without": weeks_without,
        "pct": float(pct),
    })

# ---------------------------------------------------------------------------
# 8. outliers
# ---------------------------------------------------------------------------
all_views = df["views"].astype(float)
mean_views = float(all_views.mean())
std_views = float(all_views.std(ddof=0))
threshold_viral = mean_views + 3 * std_views
threshold_under = max(0.0, mean_views - std_views)

viral_mask = df["views"] >= threshold_viral
under_mask = df["views"] <= threshold_under
viral_df = df[viral_mask].copy()

viral_genres = Counter(viral_df["game_genre"].dropna().astype(str).tolist())
viral_formats = Counter(viral_df["video_format_label"].dropna().astype(str).tolist())

# top 8 viral video records — used for HTML rendering, not stored in JSON
viral_top = (
    viral_df.sort_values("views", ascending=False)
    .head(8)[["views", "game_genre", "title"]]
    .to_dict("records")
)

outliers = {
    "threshold_viral": int(round(threshold_viral)),
    "threshold_under": int(round(threshold_under)),
    "viral_count": int(viral_mask.sum()),
    "under_count": int(under_mask.sum()),
    "viral_videos": [],
    "viral_genres": dict(viral_genres.most_common()),
    "viral_formats": dict(viral_formats.most_common()),
    "mean": int(round(mean_views)),
    "std": int(round(std_views)),
}

# ---------------------------------------------------------------------------
# 9. seasonal_genre
# ---------------------------------------------------------------------------
TOP_GENRES = ["action_adventure", "extreme_sports", "horror", "other",
              "racing", "rpg", "shooter", "sports"]

seasonal_genre: dict[str, list[dict]] = {}
for g in TOP_GENRES:
    sub = df[df["game_genre"] == g]
    months_out: list[dict] = []
    for m in range(1, 13):
        msub = sub[sub["month"] == m]
        n = int(len(msub))
        if n == 0:
            months_out.append({"month": m, "avg_views": 0, "n": 0})
        else:
            months_out.append({
                "month": m,
                "avg_views": round(float(msub["views"].mean()), 0),
                "n": n,
            })
    seasonal_genre[g] = months_out

# ---------------------------------------------------------------------------
# 10. traffic_detailed (per recent year)
# ---------------------------------------------------------------------------
traffic_detailed: list[dict] = []
for y in RECENT_YEARS:
    sub = df[df["year"] == y]
    if sub.empty:
        continue
    total_views = float(sub["views"].sum())
    browse_abs = float(sub["_browse_views"].sum())
    suggested_abs = float(sub["_suggested_views"].sum())
    search_abs = float(sub["_search_views"].sum())
    pct = lambda v: round(v / total_views * 100, 1) if total_views else 0.0
    traffic_detailed.append({
        "year": int(y),
        "browse_pct": pct(browse_abs),
        "browse_abs": int(round(browse_abs)),
        "suggested_pct": pct(suggested_abs),
        "suggested_abs": int(round(suggested_abs)),
        "search_pct": pct(search_abs),
        "search_abs": int(round(search_abs)),
        "n": int(len(sub)),
    })

# Per-year breakdown for the full history table
traffic_by_year_all: list[dict] = []
for y in YEARS_ALL:
    sub = df[df["year"] == y]
    if sub.empty:
        continue
    total_views = float(sub["views"].sum())
    browse_abs = float(sub["_browse_views"].sum())
    suggested_abs = float(sub["_suggested_views"].sum())
    search_abs = float(sub["_search_views"].sum())
    pct = lambda v: round(v / total_views * 100, 1) if total_views else 0.0
    traffic_by_year_all.append({
        "year": int(y),
        "browse_pct": pct(browse_abs),
        "browse_abs": int(round(browse_abs)),
        "suggested_pct": pct(suggested_abs),
        "suggested_abs": int(round(suggested_abs)),
        "search_pct": pct(search_abs),
        "search_abs": int(round(search_abs)),
    })

# ---------------------------------------------------------------------------
# Yearly stats for the full history conversion table (2020+)
# ---------------------------------------------------------------------------
yearly_full: list[dict] = []
prev = None
for y in YEARS_ALL:
    sub = df[df["year"] == y]
    if sub.empty:
        continue
    n = int(len(sub))
    views_total = float(sub["views"].sum())
    subs_total = float(sub["subscribers_gained"].sum())
    vpv = round(float(sub["views"].mean()), 0) if n else 0
    spk = round(subs_total / views_total * 1000, 2) if views_total else 0.0
    eng = round(float(sub["engagement_rate"].mean()), 2) if n else 0.0
    entry = {
        "year": int(y),
        "num_videos": n,
        "views_per_video": vpv,
        "subs_per_1k": spk,
        "engagement": eng,
        "vpv_trend": trend(vpv, prev["views_per_video"] if prev else None),
        "spk_trend": trend(spk, prev["subs_per_1k"] if prev else None),
        "eng_trend": trend(eng, prev["engagement"] if prev else None),
    }
    yearly_full.append(entry)
    prev = entry

# ---------------------------------------------------------------------------
# 11. interpretations
# ---------------------------------------------------------------------------
health_current = health[-1] if health else None
if health_current:
    s = health_current["score"]
    health_status = "green" if s >= 80 else ("yellow" if s >= 60 else "red")
else:
    health_status = "yellow"

best_conversion_year = max(efficiency, key=lambda e: e["subs_per_1k"]) if efficiency else None
current_conversion = efficiency[-1] if efficiency else None
best_sub_genre = sub_conversion[0] if sub_conversion else None

consistency_lookup = {c["year"]: c for c in consistency}
seasonal_peaks = {}
for g, months_out in seasonal_genre.items():
    peak = max(months_out, key=lambda m: m["avg_views"])
    if peak["n"] == 0:
        continue
    seasonal_peaks[g] = {
        "month": MONTHS_ES[peak["month"] - 1],
        "month_num": peak["month"],
        "views": peak["avg_views"],
    }

interpretations = {
    "health_current": health_current,
    "health_status": health_status,
    "best_conversion_year": best_conversion_year,
    "current_conversion": current_conversion,
    "best_sub_genre": best_sub_genre,
    f"consistency_2025": consistency_lookup.get(2025),
    f"consistency_{CURRENT_YEAR}": consistency_lookup.get(CURRENT_YEAR),
    "seasonal_peaks": seasonal_peaks,
}

# ---------------------------------------------------------------------------
# Final aggregated D dict
# ---------------------------------------------------------------------------
D = {
    "efficiency": efficiency,
    "health": health,
    "winners": winners,
    "sub_conversion": sub_conversion,
    "avd_relative": avd_relative,
    "consistency": consistency,
    "outliers": outliers,
    "seasonal_genre": seasonal_genre,
    "traffic_detailed": traffic_detailed,
    "interpretations": interpretations,
}

# ---------------------------------------------------------------------------
# Read existing file to extract the JS tail (var fmt= ... </script>)
# ---------------------------------------------------------------------------
with open(TEMPLATE_HTML, "r", encoding="utf-8") as f:
    cur = f.read()

m = re.search(r"var fmt=function.*?</script>", cur, re.DOTALL)
if not m:
    raise RuntimeError("Could not find 'var fmt=function ... </script>' in existing metrics.html")
js_tail = m.group(0)  # includes closing </script>

# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
period_label = health_current["label"] if health_current else f"{MONTHS_ES[TODAY.month-1]} {CURRENT_YEAR}"

# Health section rendering
if health_current:
    hc = health_current
    health_dato = f'<div class="dato {health_status[0]}">{hc["score"]}/100</div>'
    health_ctx = (
        f'<div class="context">Health Score ({hc["label"]}). '
        f'Freq={hc["freq"]:.0f} Eng={hc["eng"]:.0f} '
        f'AVD={hc["avd"]:.0f} Ret={hc["ret"]:.0f}</div>'
    )
    health_section_class = health_status  # "green" / "yellow" / "red"
else:
    health_dato = '<div class="dato y">--/100</div>'
    health_ctx = '<div class="context">Sin datos.</div>'
    health_section_class = "yellow"

# Quarterly conversion table rows (2025+, current quarter flagged as projected)
yearly_rows = []
for r in quarterly_efficiency:
    label = r["label"]
    cls = ' style="font-style:italic;border-left:3px dashed var(--a4)"' if r["projected"] else ''
    label_html = (
        f'<b>{label}</b> <span style="color:var(--a4);font-size:10px">(PROY &times;{r["projected_factor"]:.2f})</span>'
        if r["projected"] else f'<b>{label}</b>'
    )
    n_html = (
        f'{r["num_videos"]} <span style="color:var(--a4);font-size:10px">&rarr;{r["num_videos_projected"]}</span>'
        if r["projected"] else f'{r["num_videos"]}'
    )
    yearly_rows.append(
        f"<tr{cls}>"
        f"<td>{label_html}</td>"
        f"<td>{n_html}</td>"
        f"<td>{int(r['views_per_video']):,}</td>"
        f"{trend_td(r['vpv_trend'])}"
        f"<td>{r['subs_per_1k']:.2f}</td>"
        f"{trend_td(r['spk_trend'])}"
        f"<td>{r['engagement']:.2f}%</td>"
        f"{trend_td(r['eng_trend'])}"
        "</tr>"
    )
yearly_table_body = "".join(yearly_rows)
projected_q = next((r for r in quarterly_efficiency if r["projected"]), None)

# Conversion title (best conversion year)
if best_conversion_year and current_conversion:
    cc = current_conversion
    bcy = best_conversion_year
    conv_dato = f'<div class="dato y">{cc["subs_per_1k"]:.2f} subs/1K</div>'
    conv_ctx = f'<div class="context">Mejor: {bcy["year"]} con {bcy["subs_per_1k"]:.2f}.</div>'
else:
    conv_dato = '<div class="dato y">-- subs/1K</div>'
    conv_ctx = '<div class="context">Sin datos.</div>'

# Sub-conversion best-genre title (filtered: exclude tiny-sample genres)
EXCLUDE_TINY = {"survival", "battle_royale", "platformer"}
filtered_genres = [g for g in sub_conversion if not (g["genre"] in EXCLUDE_TINY and g["n"] < 5)]
# Always exclude survival/platformer/battle_royale for the headline title to match
# the existing "best is sports" pattern
title_pool = [g for g in sub_conversion if g["genre"] not in EXCLUDE_TINY]
top_genre_title = title_pool[0] if title_pool else (sub_conversion[0] if sub_conversion else None)
if top_genre_title:
    subconv_dato = (
        f'<div class="dato g">{top_genre_title["genre"]} = '
        f'{top_genre_title["subs_per_1k"]:.2f} subs/1K</div>'
    )
else:
    subconv_dato = '<div class="dato g">-- subs/1K</div>'

# Top 5 list for action-box
top5 = title_pool[:5] if title_pool else sub_conversion[:5]
subconv_lis = "".join(
    f'<li><b>{g["genre"]}</b>: {g["subs_per_1k"]:.2f} subs/1K ({g["total_subs"]:,} subs)</li>'
    for g in top5
)

# Consistency section
cons_current = consistency_lookup.get(CURRENT_YEAR)
cons_2025 = consistency_lookup.get(2025)
if cons_current:
    p = cons_current["pct"]
    cons_class = "g" if p >= 80 else ("y" if p >= 50 else "r")
    cons_section_class = {"g": "green", "y": "yellow", "r": "red"}[cons_class]
    cons_dato = f'<div class="dato {cons_class}">{p:.0f}% en {CURRENT_YEAR}</div>'
    parts = []
    if cons_2025 and CURRENT_YEAR != 2025:
        parts.append(f'2025: {cons_2025["weeks_without"]} semanas sin publicar.')
    parts.append(
        f'{CURRENT_YEAR}: {cons_current["weeks_with"]}/{cons_current["total_weeks"]} semanas.'
    )
    cons_ctx = f'<div class="context">{" ".join(parts)}</div>'
else:
    cons_section_class = "yellow"
    cons_dato = '<div class="dato y">-- en --</div>'
    cons_ctx = '<div class="context">Sin datos.</div>'

# Viral section
viral_dato = f'<div class="dato p">{outliers["viral_count"]} virales</div>'
viral_ctx = (
    f'<div class="context">Umbral: &gt;{outliers["threshold_viral"]:,} views.</div>'
)
viral_items = "".join(
    f'<div style="font-size:12px;color:var(--tx2);margin:4px 0;padding:6px;'
    f'background:var(--bg3);border-radius:6px">'
    f'<b style="color:var(--a4)">{int(v["views"]):,}</b> | '
    f'{v["game_genre"] if not pd.isna(v["game_genre"]) else "n/a"} | '
    f'{(str(v["title"])[:80])}'
    f'</div>'
    for v in viral_top
)

# Traffic table (full history)
traffic_rows = []
for r in traffic_by_year_all:
    traffic_rows.append(
        "<tr>"
        f"<td><b>{r['year']}</b></td>"
        f"<td>{r['browse_pct']:.1f}%</td>"
        f"<td>{r['browse_abs']:,}</td>"
        f"<td>{r['suggested_pct']:.1f}%</td>"
        f"<td>{r['suggested_abs']:,}</td>"
        f"<td>{r['search_pct']:.1f}%</td>"
        f"<td>{r['search_abs']:,}</td>"
        "</tr>"
    )
traffic_table_body = "".join(traffic_rows)
current_traffic = next((t for t in traffic_detailed if t["year"] == CURRENT_YEAR), None)
if current_traffic is None and traffic_detailed:
    current_traffic = traffic_detailed[-1]
if current_traffic:
    traffic_dato = f'<div class="dato y">{current_traffic["browse_pct"]:.1f}% Browse</div>'
else:
    traffic_dato = '<div class="dato y">--% Browse</div>'

# Seasonal action-box: top 6 genres by peak avg_views
genre_peaks_sorted = sorted(
    seasonal_peaks.items(),
    key=lambda kv: kv[1]["views"],
    reverse=True,
)[:6]
seasonal_lis = "".join(
    f'<li><b>{g}</b>: pico en <b>{p["month"]}</b> ({int(p["views"]):,} avg)</li>'
    for g, p in genre_peaks_sorted
)

# ---------------------------------------------------------------------------
# Assemble full HTML
# ---------------------------------------------------------------------------
if projected_q:
    header_period = (
        f"Datos auditados &middot; Per&iacute;odo: 2025Q1 - {projected_q['label']} "
        f"<span style=\"color:var(--a4)\">(en curso, proyectado &times;{projected_q['projected_factor']:.2f} | "
        f"{projected_q['days_elapsed']}/{projected_q['days_total']} d&iacute;as)</span>"
    )
else:
    header_period = f"Datos auditados &middot; Per&iacute;odo: Enero 2025 - {period_label}"

html = (
    '<!DOCTYPE html><html lang="es"><head>'
    '<meta charset="UTF-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1.0">'
    '<title>Joy Of Gaming &mdash; Informe Ejecutivo</title>'
    '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>'
    '<style>:root{--bg:#0E1117;--bg2:#161B22;--bg3:#1C2333;--a1:#FF4444;--a2:#00B894;'
    '--a3:#6C5CE7;--a4:#FDCB6E;--a5:#00CEC9;--tx:#E6EDF3;--tx2:#8B949E;--bd:#30363D}'
    '*{margin:0;padding:0;box-sizing:border-box}'
    'body{background:var(--bg);color:var(--tx);font-family:"Segoe UI",system-ui,sans-serif;line-height:1.7}'
    '.hdr{background:linear-gradient(135deg,#0a0a1a,#1a1a2e 50%,#16213e);padding:28px 32px;'
    'border-bottom:3px solid var(--a3);text-align:center}'
    '.hdr h1{font-size:24px;font-weight:800;color:var(--a3)}'
    '.hdr .sub{color:var(--tx2);font-size:12px;margin-top:4px}'
    '.hdr a{color:var(--a4);font-size:12px;text-decoration:none;display:inline-block;margin-top:6px}'
    '.wrap{max-width:1000px;margin:0 auto;padding:24px}'
    '.question{font-size:20px;font-weight:800;color:var(--tx);margin:32px 0 6px}'
    '.section{border-radius:12px;padding:20px;margin-bottom:24px;border-left:5px solid;background:var(--bg2)}'
    '.section.green{border-color:var(--a2)}'
    '.section.yellow{border-color:var(--a4)}'
    '.section.red{border-color:var(--a1)}'
    '.dato{font-size:36px;font-weight:800;margin:8px 0}'
    '.dato.g{color:var(--a2)}.dato.y{color:var(--a4)}.dato.r{color:var(--a1)}.dato.p{color:var(--a3)}'
    '.context{font-size:13px;color:var(--tx2);margin-bottom:12px;line-height:1.8}'
    '.action-box{background:rgba(0,184,148,.1);border:1px solid var(--a2);border-radius:10px;padding:14px 18px;margin-top:12px}'
    '.action-box h4{color:var(--a2);font-size:13px;margin-bottom:6px;text-transform:uppercase;letter-spacing:1px}'
    '.action-box p,.action-box li{font-size:13px;line-height:1.8}'
    '.action-box ul{padding-left:18px}'
    '.cb{position:relative;height:280px;margin:12px 0}'
    'table{width:100%;border-collapse:collapse;font-size:12px;margin:12px 0}'
    'th{background:var(--bg3);color:var(--tx2);padding:8px;text-align:left;font-size:10px;text-transform:uppercase}'
    'td{padding:7px 8px;border-bottom:1px solid var(--bd)}'
    'tr:hover td{background:var(--bg3)}'
    '.pos{color:var(--a2)}.neg{color:var(--a1)}'
    '.summary{background:linear-gradient(135deg,#0a1628,#162340);border:2px solid var(--a4);border-radius:14px;padding:24px;margin-top:32px}'
    '.summary h2{color:var(--a4);font-size:18px;margin-bottom:14px;text-align:center}'
    '.summary ol{padding-left:20px}'
    '.summary li{font-size:14px;margin-bottom:10px;line-height:1.7}'
    '.summary li b{color:var(--a4)}'
    '</style></head><body>'

    '<div class="hdr">'
    '<h1>Informe Ejecutivo &mdash; Joy Of Gaming</h1>'
    f'<div class="sub">{header_period}</div>'
    '<a href="strategic_dashboard.html">Volver al Dashboard</a>'
    '</div>'

    '<div class="wrap">'

    # Q1: Health
    '<div class="question">Esta mejorando la calidad de mi contenido?</div>'
    f'<div class="section {health_section_class}">'
    f'{health_dato}'
    f'{health_ctx}'
    '<div class="cb"><canvas id="c-health"></canvas></div>'
    '<div class="action-box"><h4>Que hacer</h4><ul>'
    '<li>Minimo <b>3 videos/semana</b></li>'
    '<li>AVD sobre <b>90 segundos</b></li>'
    '<li>Retencion 30s sobre <b>65%</b></li>'
    '</ul></div>'
    '</div>'

    # Q2: Conversion
    '<div class="question">Estoy convirtiendo views en comunidad?</div>'
    '<div class="section yellow">'
    f'{conv_dato}'
    f'{conv_ctx}'
    '<table><thead><tr>'
    '<th>Quarter</th><th>Videos</th><th>Views/Vid</th><th></th>'
    '<th>Subs/1K</th><th></th><th>Eng%</th><th></th>'
    '</tr></thead><tbody>'
    f'{yearly_table_body}'
    '</tbody></table>'
    '<div class="action-box"><h4>Que hacer</h4>'
    '<p>Cada view vale mas hoy. <b>Calidad con PS5 Pro</b> sobre cantidad.</p>'
    '</div></div>'

    # Q3: Sub conversion by genre
    '<div class="question">Que contenido construye mi canal?</div>'
    '<div class="section green">'
    f'{subconv_dato}'
    '<div class="cb"><canvas id="c-subconv"></canvas></div>'
    '<div class="action-box"><h4>Que hacer</h4><ul>'
    f'{subconv_lis}'
    '</ul></div></div>'

    # Q4: Consistency
    '<div class="question">Estoy publicando suficiente?</div>'
    f'<div class="section {cons_section_class}">'
    f'{cons_dato}'
    f'{cons_ctx}'
    '<div class="cb"><canvas id="c-consist"></canvas></div>'
    '<div class="action-box"><h4>Que hacer</h4>'
    '<p><b>3 videos/semana sin faltar.</b> Inconsistencia = -67% crecimiento.</p>'
    '</div></div>'

    # Q5: Virals
    '<div class="question">Que videos se viralizan?</div>'
    '<div class="section yellow">'
    f'{viral_dato}'
    f'{viral_ctx}'
    '<div style="margin:12px 0">'
    f'{viral_items}'
    '</div>'
    '<div class="action-box"><h4>Que hacer</h4>'
    '<p>Virales = <b>racing + extreme_sports</b> con <b>FIRST PERSON / INSANE</b>. Publicar 1/semana asi.</p>'
    '</div></div>'

    # Q6: Traffic
    '<div class="question">YouTube me esta recomendando?</div>'
    '<div class="section yellow">'
    f'{traffic_dato}'
    '<table><thead><tr>'
    '<th>A&ntilde;o</th><th>Browse %</th><th>Vol</th>'
    '<th>Suggested %</th><th>Vol</th>'
    '<th>Search %</th><th>Vol</th>'
    '</tr></thead><tbody>'
    f'{traffic_table_body}'
    '</tbody></table>'
    '<div class="action-box"><h4>Que hacer</h4>'
    '<p>Browse alto pero volumen bajo. <b>Publicar mas</b> + series de 3-5 videos.</p>'
    '</div></div>'

    # Q7: Seasonal
    '<div class="question">Hay estacionalidad?</div>'
    '<div class="section green">'
    '<div class="cb" style="height:320px"><canvas id="c-seasonal"></canvas></div>'
    '<div class="action-box"><h4>Que hacer</h4><ul>'
    f'{seasonal_lis}'
    '</ul></div></div>'

    # Static summary
    '<div class="summary">'
    '<h2>5 Acciones Para Esta Semana</h2>'
    '<ol>'
    '<li><b>3 videos largos + 3 shorts sin excepcion</b></li>'
    '<li><b>1 video First Person POV de racing/extreme sports</b></li>'
    '<li><b>Titulos: juego + PS5 + Ultra Realistic + 4K HDR</b></li>'
    '<li><b>Serie de 3 videos del mismo juego</b></li>'
    '<li><b>Thumbnail con modo foto PS5 sin texto</b></li>'
    '</ol>'
    '</div>'

    '</div>'  # /wrap

    # JS block
    '<script>\n'
    'var D=' + json.dumps(D, ensure_ascii=False, separators=(",", ":")) + ';\n'
    + js_tail +
    '</body></html>'
)

# ---------------------------------------------------------------------------
# Save + summary
# ---------------------------------------------------------------------------
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

size = os.path.getsize(OUTPUT_HTML)
period_str = f"Ene 2025 - {period_label}"
hc_str = (
    f"{health_current['label']} score={health_current['score']}"
    if health_current else "n/a"
)
print(f"metrics.html rebuilt: {size:,} bytes | Period: {period_str} | Latest health: {hc_str}")
