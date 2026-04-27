"""Weekly Auditor Agent — Joy Of Gaming.

Analiza videos publicados la semana anterior (Lunes-Domingo Chile),
compara contra baseline normalizado por tiempo (últimos 90 días),
genera reporte profesional y lo envía por email.

Output:
  - weekly_audits/audit_YYYY-MM-DD_YYYY-MM-DD.md (markdown)
  - strategic_data.json key 'weekly_audit' (para dashboard tab)
  - Email vía Resend a EMAIL_TO

Run:
  python weekly_auditor.py
"""
import os
import sys
import json
import pickle
import base64
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHILE_TZ = timezone(timedelta(hours=-4))
TOKEN_FILE = "token.pickle"

# `os.environ.get(..., default)` returns "" (no el default) cuando el secret existe
# pero está vacío en GitHub Actions. Usamos `or` para cubrir ambos casos.
RESEND_API_KEY = os.environ.get("RESEND_API_KEY") or ""
EMAIL_TO = os.environ.get("EMAIL_TO") or "bcorreamac@gmail.com"
EMAIL_FROM = os.environ.get("EMAIL_FROM") or "Joy Of Gaming <onboarding@resend.dev>"

DASHBOARD_URL = (
    os.environ.get("DASHBOARD_URL")
    or "https://bcorreamac-cpu.github.io/yt-gaming-analytics/strategic_dashboard.html"
)

# Override para correr el auditor con un rango histórico en lugar de la semana pasada.
CUSTOM_START_DATE = (os.environ.get("CUSTOM_START_DATE") or "").strip()
CUSTOM_END_DATE = (os.environ.get("CUSTOM_END_DATE") or "").strip()

BASELINE_DAYS = 90       # ventana del baseline
EXCLUDE_RECENT_DAYS = 14 # excluir videos muy nuevos del baseline

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def get_services():
    if not Path(TOKEN_FILE).exists():
        sys.exit(f"ERROR: {TOKEN_FILE} no existe. Re-autoriza con data_collector.py.")
    with open(TOKEN_FILE, "rb") as f:
        creds = pickle.load(f)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(TOKEN_FILE, "wb") as f:
            pickle.dump(creds, f)
    yt = build("youtube", "v3", credentials=creds)
    analytics = build("youtubeAnalytics", "v2", credentials=creds)
    ch = yt.channels().list(part="id", mine=True).execute()
    return yt, analytics, ch["items"][0]["id"]


# ---------------------------------------------------------------------------
# Date math
# ---------------------------------------------------------------------------

def previous_week_range(now_chile=None):
    """Devuelve (lunes 00:00, domingo 23:59) de la semana anterior en Chile time.

    Si CUSTOM_START_DATE y CUSTOM_END_DATE están seteadas, las usa en lugar
    de calcular automáticamente. Formato: YYYY-MM-DD.
    """
    if CUSTOM_START_DATE and CUSTOM_END_DATE:
        start = datetime.strptime(CUSTOM_START_DATE, "%Y-%m-%d").replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=CHILE_TZ)
        end = datetime.strptime(CUSTOM_END_DATE, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, microsecond=0, tzinfo=CHILE_TZ)
        print(f"[custom range] {start.date()} → {end.date()}")
        return start, end

    now = now_chile or datetime.now(CHILE_TZ)
    # weekday(): Monday=0 ... Sunday=6
    days_since_monday = now.weekday()
    this_monday = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
    last_monday = this_monday - timedelta(days=7)
    last_sunday = last_monday + timedelta(days=6, hours=23, minutes=59, seconds=59)
    return last_monday, last_sunday


# ---------------------------------------------------------------------------
# Fetch videos published in the week
# ---------------------------------------------------------------------------

def get_uploads_playlist(yt, channel_id):
    r = yt.channels().list(part="contentDetails", id=channel_id).execute()
    return r["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


def get_videos_in_range(yt, uploads_playlist, start_utc, end_utc):
    """Devuelve lista de video IDs públicos publicados en [start_utc, end_utc]."""
    ids = []
    page_token = None
    keep_paging = True
    while keep_paging:
        resp = yt.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_playlist,
            maxResults=50,
            pageToken=page_token,
        ).execute()
        for item in resp.get("items", []):
            pub_iso = item["contentDetails"].get("videoPublishedAt")
            if not pub_iso:
                continue
            pub = datetime.fromisoformat(pub_iso.replace("Z", "+00:00"))
            if pub < start_utc:
                keep_paging = False
                break
            if pub <= end_utc:
                ids.append(item["contentDetails"]["videoId"])
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return ids


def get_video_details(yt, video_ids):
    """Devuelve detalles + filtra no-públicos."""
    if not video_ids:
        return []
    out = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        resp = yt.videos().list(
            part="snippet,contentDetails,statistics,status",
            id=",".join(batch),
        ).execute()
        for it in resp.get("items", []):
            status = it.get("status", {})
            if status.get("privacyStatus") != "public":
                continue
            if status.get("uploadStatus") != "processed":
                continue
            sn = it["snippet"]
            stats = it.get("statistics", {})
            duration = parse_duration(it["contentDetails"].get("duration", "PT0S"))
            if duration < 60:  # skip shorts
                continue
            out.append({
                "video_id": it["id"],
                "title": sn.get("title", ""),
                "published_at": sn.get("publishedAt", ""),
                "duration_sec": duration,
                "thumbnail": sn.get("thumbnails", {}).get("high", {}).get("url", ""),
                "tags": sn.get("tags", []),
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0)),
            })
    return out


def parse_duration(iso):
    import re
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    if not m:
        return 0
    h, mi, s = m.groups(default="0")
    return int(h) * 3600 + int(mi) * 60 + int(s)


# ---------------------------------------------------------------------------
# Analytics metrics per video
# ---------------------------------------------------------------------------

def get_video_metrics(analytics, channel_id, video_id, start_date, end_date):
    """Pull AVD, retention, CTR, etc. desde Analytics API."""
    out = {}

    # Q1: core engagement metrics
    try:
        r = analytics.reports().query(
            ids=f"channel=={channel_id}",
            startDate=start_date,
            endDate=end_date,
            metrics=("views,estimatedMinutesWatched,averageViewDuration,"
                     "averageViewPercentage,subscribersGained,likes,comments,shares"),
            filters=f"video=={video_id}",
        ).execute()
        if r.get("rows"):
            headers = [c["name"] for c in r["columnHeaders"]]
            out.update(dict(zip(headers, r["rows"][0])))
    except HttpError as e:
        print(f"  [warn] metrics q1 {video_id}: {e.resp.status}")

    # Q2 individual: omitido — `impressions` no acepta `filters=video==X`.
    # En su lugar usamos `fetch_impressions_bulk()` después del loop principal.

    # Q3: retention curve (primer 30s)
    try:
        r = analytics.reports().query(
            ids=f"channel=={channel_id}",
            startDate=start_date,
            endDate=end_date,
            metrics="audienceWatchRatio",
            dimensions="elapsedVideoTimeRatio",
            filters=f"video=={video_id}",
        ).execute()
        rows = r.get("rows", [])
        if rows:
            # rows = [[ratio_pos, watch_ratio], ...]
            # retention_30s = watch ratio at the 30s mark approximately
            sorted_rows = sorted(rows, key=lambda x: x[0])
            # Find closest to 30s/duration
            out["_retention_curve"] = sorted_rows
    except HttpError:
        pass

    return out


def fetch_impressions_bulk(analytics, channel_id, start_date, end_date, max_results=200):
    """Bulk fetch de impressions+CTR via dimensions=video.

    `impressions` no permite `filters=video==X` (error 400). En su lugar,
    se pide la lista por dimensión y luego se matchea por video_id.
    Retorna dict {video_id: {"impressions": int, "ctr": float}}.
    """
    out = {}
    try:
        r = analytics.reports().query(
            ids=f"channel=={channel_id}",
            startDate=start_date,
            endDate=end_date,
            metrics="impressions,impressionsClickThroughRate",
            dimensions="video",
            sort="-impressions",
            maxResults=max_results,
        ).execute()
        for row in r.get("rows", []):
            vid = row[0]
            out[vid] = {
                "impressions": int(row[1] or 0),
                "impressionsClickThroughRate": float(row[2] or 0),
            }
    except HttpError as e:
        print(f"  [warn] bulk impressions fetch: {e.resp.status} "
              f"{(e.content or b'')[:200]}")
    return out


# ---------------------------------------------------------------------------
# Baseline (last 90 days, normalized)
# ---------------------------------------------------------------------------

def build_baseline(df_all, today, exclude_video_ids=None):
    cutoff_low = today - timedelta(days=BASELINE_DAYS)
    cutoff_high = today - timedelta(days=EXCLUDE_RECENT_DAYS)
    base = df_all[(df_all["published_at"] >= cutoff_low) &
                  (df_all["published_at"] <= cutoff_high)].copy()
    if exclude_video_ids:
        base = base[~base["video_id"].isin(exclude_video_ids)]
    base["days_old"] = (today - base["published_at"]).dt.days.clip(lower=1)
    base["views_per_day"] = base["views"] / base["days_old"]
    return base


def find_peers(video, baseline_df):
    """Encuentra cohorte comparable: mismo juego (n>=3) > genre+formato (n>=5) > canal."""
    g = video.get("game_name") or ""
    if g:
        same_game = baseline_df[baseline_df["game_name"] == g]
        if len(same_game) >= 3:
            return same_game, f"mismo juego ({g})"
    same_gf = baseline_df[
        (baseline_df["game_genre"] == video.get("game_genre")) &
        (baseline_df["video_format"] == video.get("video_format"))
    ]
    if len(same_gf) >= 5:
        return same_gf, f"mismo género+formato ({video.get('game_genre')}/{video.get('video_format')})"
    return baseline_df, "canal últimos 90 días"


def score_video(video, peers, today):
    """Calcula deltas vs baseline para cada métrica clave."""
    days = max(1, (today - video["published_at"]).days)
    video["days_old"] = days
    video["views_per_day"] = video["views"] / days

    peer_med_vpd = peers["views_per_day"].median() or 1
    peer_med_avd = peers["avg_view_duration_sec"].median() if "avg_view_duration_sec" in peers.columns else 0
    peer_med_avp = peers["avg_view_percentage"].median() if "avg_view_percentage" in peers.columns else 0
    peer_med_eng = peers["engagement_rate"].median() if "engagement_rate" in peers.columns else 0
    peer_med_ctr = peers["ctr"].median() if "ctr" in peers.columns and peers["ctr"].sum() > 0 else 0
    peer_med_r30 = peers["retention_30s"].median() if "retention_30s" in peers.columns else 0

    score = {
        "peer_n": len(peers),
        "views_per_day": round(video["views_per_day"], 1),
        "peer_views_per_day": round(peer_med_vpd, 1),
        "vpd_delta_pct": round((video["views_per_day"] / peer_med_vpd - 1) * 100, 0) if peer_med_vpd > 0 else 0,
        "avd": round(video.get("averageViewDuration", 0), 0),
        "peer_avd": round(peer_med_avd, 0),
        "avd_delta_pct": round((video.get("averageViewDuration", 0) / peer_med_avd - 1) * 100, 0) if peer_med_avd > 0 else 0,
        "avp": round(video.get("averageViewPercentage", 0), 1),
        "peer_avp": round(peer_med_avp, 1),
        "ctr": round(video.get("impressionsClickThroughRate", 0), 2),
        "peer_ctr": round(peer_med_ctr, 2),
        "retention_30s": round(video.get("_retention_30s", 0), 1),
        "peer_r30": round(peer_med_r30, 1),
        "engagement": round((video.get("likes", 0) + video.get("comments", 0)) / max(video["views"], 1) * 100, 2),
        "peer_engagement": round(peer_med_eng, 2),
    }
    return score


# ---------------------------------------------------------------------------
# Saturation detector
# ---------------------------------------------------------------------------

def detect_saturation(df_all, weekly_videos, today):
    cutoff = today - timedelta(days=30)
    last_30 = df_all[df_all["published_at"] >= cutoff].copy().sort_values("published_at", ascending=False)
    warnings = []
    games_in_week = {v.get("game_name") for v in weekly_videos if v.get("game_name")}
    for game in games_in_week:
        sg = last_30[last_30["game_name"] == game]
        if len(sg) <= 4:
            continue
        first_half = sg.head(len(sg) // 2)
        second_half = sg.tail(len(sg) // 2)
        if first_half["views"].mean() == 0:
            continue
        decay = (first_half["views"].mean() - second_half["views"].mean()) / second_half["views"].mean() * 100
        warnings.append({
            "game": game,
            "count_30d": int(len(sg)),
            "decay_pct": round(decay, 0),
        })
    return [w for w in warnings if w["count_30d"] >= 5]


# ---------------------------------------------------------------------------
# Qualitative analysis per video
# ---------------------------------------------------------------------------

def analyze_video(video, score):
    good, bad, suggestions = [], [], []

    if score["vpd_delta_pct"] >= 50:
        good.append(f"📈 Views/día {score['vpd_delta_pct']:+.0f}% sobre cohorte (peer median: {score['peer_views_per_day']}/d)")
    elif score["vpd_delta_pct"] <= -30:
        bad.append(f"📉 Views/día {score['vpd_delta_pct']:+.0f}% bajo cohorte (peer median: {score['peer_views_per_day']}/d)")

    ctr = score["ctr"]
    if ctr > 0:
        if ctr >= 6:
            good.append(f"🎯 CTR {ctr}% (excelente, sobre 6%)")
        elif ctr <= 3:
            bad.append(f"🎯 CTR {ctr}% (bajo; peer median: {score['peer_ctr']}%)")
            suggestions.append("Probar A/B test del thumbnail (rostro + texto contraste)")

    avp = score["avp"]
    if avp >= 50:
        good.append(f"⏱️ Retención promedio {avp}% (sólida)")
    elif avp <= 35 and avp > 0:
        bad.append(f"⏱️ Retención promedio {avp}% (baja; viewers se van pronto)")
        suggestions.append("Revisar primeros 30s — agregar hook claro o cinemático fuerte")

    eng = score["engagement"]
    peer_eng = score["peer_engagement"]
    if peer_eng > 0 and eng < peer_eng * 0.6:
        bad.append(f"💬 Engagement {eng}% (peer: {peer_eng}%)")
        suggestions.append("Pedir like/comment explícito en el video")

    return good, bad, suggestions


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------

def compute_aggregates(videos, scores, baseline):
    if not videos:
        return {}
    def safe_mean(vals):
        vals = [v for v in vals if v and v > 0]
        return float(np.mean(vals)) if vals else 0.0

    total_views = sum(v["views"] for v in videos)
    total_subs = sum(v.get("subscribersGained", 0) for v in videos)
    avg_avd = safe_mean([s["avd"] for s in scores])
    avg_avp = safe_mean([s["avp"] for s in scores])
    avg_ctr = safe_mean([s["ctr"] for s in scores])
    avg_eng = safe_mean([s["engagement"] for s in scores])
    avg_vpd = safe_mean([s["views_per_day"] for s in scores])
    peer_avg_vpd = baseline["views_per_day"].median() if len(baseline) else 0
    if pd.isna(peer_avg_vpd):
        peer_avg_vpd = 0

    return {
        "video_count": len(videos),
        "total_views": int(total_views),
        "total_subs_gained": int(total_subs),
        "avg_avd": round(avg_avd, 0),
        "avg_avp": round(avg_avp, 1),
        "avg_ctr": round(avg_ctr, 2),
        "avg_engagement": round(avg_eng, 2),
        "avg_views_per_day": round(avg_vpd, 1),
        "baseline_avg_views_per_day": round(peer_avg_vpd, 1),
        "baseline_n": len(baseline),
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_html_email(start, end, agg, items, saturation, recommended):
    """Email HTML — inline styles para compatibilidad con clientes de mail."""
    bg = "#0E1117"
    card = "#161B22"
    bd = "#30363D"
    tx = "#E6EDF3"
    tx2 = "#8B949E"
    a1 = "#FF4444"; a2 = "#00B894"; a3 = "#6C5CE7"; a4 = "#FDCB6E"; a5 = "#00CEC9"

    def kpi(label, val, delta_pct=None, color=a5):
        delta_html = ""
        if delta_pct is not None:
            sign = "+" if delta_pct >= 0 else ""
            dcolor = a2 if delta_pct >= 0 else a1
            delta_html = f'<div style="color:{dcolor};font-size:12px;margin-top:4px">{sign}{delta_pct:.0f}% vs baseline</div>'
        return f'''
        <td style="background:{card};border:1px solid {bd};border-radius:8px;padding:14px;width:25%;vertical-align:top">
          <div style="color:{tx2};font-size:11px;text-transform:uppercase;letter-spacing:.5px">{label}</div>
          <div style="color:{color};font-size:22px;font-weight:700;margin-top:6px">{val}</div>
          {delta_html}
        </td>'''

    delta_views = None
    if agg.get("baseline_avg_views_per_day", 0) > 0:
        delta_views = (agg["avg_views_per_day"] / agg["baseline_avg_views_per_day"] - 1) * 100

    kpi_row = f'''
    <table cellpadding="0" cellspacing="6" style="width:100%;margin:18px 0">
      <tr>
        {kpi("Videos publicados", agg.get("video_count", 0), color=a3)}
        {kpi("Views totales", f'{agg.get("total_views",0):,}', color=a1)}
        {kpi("Views/día prom.", agg.get("avg_views_per_day", 0), delta_views, color=a4)}
        {kpi("Subs ganados", agg.get("total_subs_gained", 0), color=a2)}
      </tr>
      <tr>
        {kpi("AVD prom.", f'{agg.get("avg_avd",0):.0f}s', color=a5)}
        {kpi("Retención prom.", f'{agg.get("avg_avp",0)}%', color=a3)}
        {kpi("CTR prom.", f'{agg.get("avg_ctr",0)}%' if agg.get("avg_ctr") else "N/D", color=a4)}
        {kpi("Engagement prom.", f'{agg.get("avg_engagement",0)}%', color=a2)}
      </tr>
    </table>'''

    # Tabla por video
    rows = []
    for it in items:
        v = it["video"]; s = it["score"]
        status = "🟢" if s["vpd_delta_pct"] >= 30 else ("🔴" if s["vpd_delta_pct"] <= -30 else "🟡")
        ctr_disp = f'{s["ctr"]}%' if s["ctr"] else "N/D"
        rows.append(f'''
        <tr style="border-bottom:1px solid {bd}">
          <td style="padding:10px;color:{tx};font-size:12px;max-width:280px">
            <a href="https://youtube.com/watch?v={v["video_id"]}" style="color:{a5};text-decoration:none">{v["title"][:70]}</a>
          </td>
          <td style="padding:10px;color:{tx2};font-size:12px;text-align:center">{v["days_old"]}d</td>
          <td style="padding:10px;color:{tx};font-size:12px;text-align:right">{v["views"]:,}</td>
          <td style="padding:10px;font-size:12px;text-align:right;color:{a2 if s["vpd_delta_pct"]>=0 else a1}">{s["vpd_delta_pct"]:+.0f}%</td>
          <td style="padding:10px;color:{tx};font-size:12px;text-align:right">{ctr_disp}</td>
          <td style="padding:10px;color:{tx};font-size:12px;text-align:right">{s["avp"]}%</td>
          <td style="padding:10px;font-size:14px;text-align:center">{status}</td>
        </tr>''')

    table_html = f'''
    <table style="width:100%;background:{card};border:1px solid {bd};border-radius:8px;border-collapse:collapse;margin:14px 0">
      <thead>
        <tr style="background:{bg}">
          <th style="text-align:left;padding:10px;color:{tx2};font-size:11px;text-transform:uppercase">Video</th>
          <th style="padding:10px;color:{tx2};font-size:11px;text-transform:uppercase">Edad</th>
          <th style="padding:10px;color:{tx2};font-size:11px;text-transform:uppercase;text-align:right">Views</th>
          <th style="padding:10px;color:{tx2};font-size:11px;text-transform:uppercase;text-align:right">Δ Views/día</th>
          <th style="padding:10px;color:{tx2};font-size:11px;text-transform:uppercase;text-align:right">CTR</th>
          <th style="padding:10px;color:{tx2};font-size:11px;text-transform:uppercase;text-align:right">Retención</th>
          <th style="padding:10px;color:{tx2};font-size:11px;text-transform:uppercase;text-align:center">Status</th>
        </tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>''' if rows else f'<p style="color:{tx2}">No hubo videos publicados en este rango.</p>'

    # Lo que funcionó / no funcionó
    good_block = []
    bad_block = []
    suggestions_block = []
    for it in items:
        for g in it["good"]:
            good_block.append(f'<li style="color:{tx};margin:6px 0"><b style="color:{a2}">{it["video"]["title"][:50]}</b>: {g}</li>')
        for b in it["bad"]:
            bad_block.append(f'<li style="color:{tx};margin:6px 0"><b style="color:{a1}">{it["video"]["title"][:50]}</b>: {b}</li>')
        for s in it["suggestions"]:
            suggestions_block.append(f'<li style="color:{tx};margin:6px 0">📌 {it["video"]["title"][:40]}: {s}</li>')

    sat_block = ""
    if saturation:
        items_sat = "".join(
            f'<li style="color:{tx};margin:6px 0"><b style="color:{a4}">{w["game"]}</b>: {w["count_30d"]} videos en 30d, decay {w["decay_pct"]:+.0f}%. Considera pausar 1-2 semanas.</li>'
            for w in saturation
        )
        sat_block = f'''
        <h3 style="color:{a4};margin-top:24px;border-bottom:2px solid {a4};padding-bottom:6px">⚠️ Alertas de Saturación</h3>
        <ul style="padding-left:20px">{items_sat}</ul>'''

    rec_block = ""
    if recommended:
        items_rec = "".join(f'<li style="color:{tx};margin:8px 0">{r}</li>' for r in recommended)
        rec_block = f'''
        <h3 style="color:{a3};margin-top:24px;border-bottom:2px solid {a3};padding-bottom:6px">🎯 Acciones Recomendadas para Esta Semana</h3>
        <ol style="padding-left:20px">{items_rec}</ol>'''

    return f'''<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:{bg};font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
<div style="max-width:780px;margin:0 auto;padding:24px;background:{bg};color:{tx}">
  <div style="border-bottom:3px solid {a1};padding-bottom:12px;margin-bottom:18px">
    <h1 style="color:{tx};margin:0;font-size:22px">📊 Auditoría Semanal — Joy Of Gaming</h1>
    <p style="color:{tx2};margin:4px 0 0;font-size:13px">Semana del {start.strftime("%d %b")} al {end.strftime("%d %b %Y")} · Generado {datetime.now(CHILE_TZ).strftime("%d %b %H:%M Chile")}</p>
  </div>

  {kpi_row}

  <h3 style="color:{a5};margin-top:24px;border-bottom:2px solid {a5};padding-bottom:6px">📋 Detalle por Video</h3>
  {table_html}

  {f'<h3 style="color:{a2};margin-top:24px;border-bottom:2px solid {a2};padding-bottom:6px">✅ Lo que funcionó</h3><ul style="padding-left:20px">{"".join(good_block) if good_block else f"<li style=color:{tx2}>Sin highlights destacados esta semana.</li>"}</ul>' if items else ""}

  {f'<h3 style="color:{a1};margin-top:24px;border-bottom:2px solid {a1};padding-bottom:6px">❌ Lo que pudo ser mejor</h3><ul style="padding-left:20px">{"".join(bad_block) if bad_block else f"<li style=color:{tx2}>Sin issues críticos esta semana.</li>"}</ul>' if items else ""}

  {sat_block}

  {f'<h3 style="color:{a4};margin-top:24px;border-bottom:2px solid {a4};padding-bottom:6px">💡 Sugerencias específicas</h3><ul style="padding-left:20px">{"".join(suggestions_block)}</ul>' if suggestions_block else ""}

  {rec_block}

  <div style="margin-top:32px;padding-top:16px;border-top:1px solid {bd};color:{tx2};font-size:12px">
    <p>Reporte automatizado · <a href="{DASHBOARD_URL}" style="color:{a5}">Ver dashboard completo</a></p>
    <p style="font-size:11px;color:{tx2}">Baseline: últimos {BASELINE_DAYS} días excluyendo videos &lt;{EXCLUDE_RECENT_DAYS}d</p>
  </div>
</div>
</body></html>'''


def render_markdown(start, end, agg, items, saturation, recommended):
    md = [f"# Auditoría Semanal — {start.strftime('%d %b')} al {end.strftime('%d %b %Y')}\n"]
    md.append(f"_Generado: {datetime.now(CHILE_TZ).strftime('%Y-%m-%d %H:%M Chile')}_\n\n")

    md.append("## Resumen\n")
    md.append(f"- Videos publicados: **{agg.get('video_count',0)}**")
    md.append(f"- Views totales: **{agg.get('total_views',0):,}**")
    md.append(f"- Views/día promedio: **{agg.get('avg_views_per_day',0)}** (baseline: {agg.get('baseline_avg_views_per_day',0)})")
    md.append(f"- Subs ganados: **{agg.get('total_subs_gained',0)}**")
    md.append(f"- AVD: **{agg.get('avg_avd',0):.0f}s** | Retención: **{agg.get('avg_avp',0)}%** | CTR: **{agg.get('avg_ctr',0)}%** | Eng: **{agg.get('avg_engagement',0)}%**\n")

    if items:
        md.append("\n## Detalle por video\n")
        md.append("| Video | Edad | Views | Δ Views/día | CTR | Retención | Status |")
        md.append("|---|---|---|---|---|---|---|")
        for it in items:
            v = it["video"]; s = it["score"]
            status = "🟢" if s["vpd_delta_pct"] >= 30 else ("🔴" if s["vpd_delta_pct"] <= -30 else "🟡")
            ctr_d = f'{s["ctr"]}%' if s["ctr"] else "N/D"
            md.append(f'| [{v["title"][:60]}](https://youtube.com/watch?v={v["video_id"]}) | {v["days_old"]}d | {v["views"]:,} | {s["vpd_delta_pct"]:+.0f}% | {ctr_d} | {s["avp"]}% | {status} |')

        md.append("\n## ✅ Lo que funcionó\n")
        any_good = False
        for it in items:
            for g in it["good"]:
                md.append(f"- **{it['video']['title'][:50]}**: {g}")
                any_good = True
        if not any_good:
            md.append("- _Sin highlights destacados esta semana._")

        md.append("\n## ❌ Lo que pudo ser mejor\n")
        any_bad = False
        for it in items:
            for b in it["bad"]:
                md.append(f"- **{it['video']['title'][:50]}**: {b}")
                any_bad = True
        if not any_bad:
            md.append("- _Sin issues críticos._")

    if saturation:
        md.append("\n## ⚠️ Alertas de saturación\n")
        for w in saturation:
            md.append(f"- **{w['game']}**: {w['count_30d']} videos en 30d, decay {w['decay_pct']:+.0f}%")

    if recommended:
        md.append("\n## 🎯 Acciones para esta semana\n")
        for r in recommended:
            md.append(f"- {r}")

    return "\n".join(md)


# ---------------------------------------------------------------------------
# Recommended actions (heurísticas)
# ---------------------------------------------------------------------------

def build_recommendations(agg, items, saturation):
    recs = []

    n = agg.get("video_count", 0)
    if n == 0:
        recs.append("Sin publicaciones esta semana — apuntar a mínimo 3 videos esta semana para mantener consistencia algorítmica.")
    elif n < 3:
        recs.append(f"Solo {n} video(s) esta semana. Apuntar a 3-5 videos para mantener frecuencia.")

    if agg.get("avg_ctr", 0) > 0 and agg["avg_ctr"] < 3:
        recs.append(f"CTR promedio {agg['avg_ctr']}% (objetivo: 4-6%). Iterar thumbnails con caras + texto contraste.")
    if agg.get("avg_avp", 0) > 0 and agg["avg_avp"] < 35:
        recs.append(f"Retención promedio {agg['avg_avp']}% (baja). Hook más fuerte en primeros 15s, evitar intros largas.")

    for w in saturation:
        recs.append(f"Pausar **{w['game']}** mínimo 1-2 semanas (decay {w['decay_pct']:+.0f}% tras {w['count_30d']} videos).")

    # Mejores videos como referencia
    winners = [it for it in items if it["score"]["vpd_delta_pct"] >= 50]
    if winners:
        top = sorted(winners, key=lambda x: x["score"]["vpd_delta_pct"], reverse=True)[0]
        recs.append(f"**Replicar fórmula**: '{top['video']['title'][:60]}' rindió {top['score']['vpd_delta_pct']:+.0f}%. Considerar otro video del mismo juego/formato.")

    losers = [it for it in items if it["score"]["vpd_delta_pct"] <= -50 and it["score"]["ctr"] > 0 and it["score"]["ctr"] < 3]
    for it in losers:
        recs.append(f"Re-thumbnail '{it['video']['title'][:50]}' (CTR {it['score']['ctr']}%, views/día {it['score']['vpd_delta_pct']:+.0f}%).")

    if not recs:
        recs.append("Semana en línea con baseline. Mantener la estrategia actual y monitorear semana próxima.")

    return recs


# ---------------------------------------------------------------------------
# Email send via Resend
# ---------------------------------------------------------------------------

def send_email(subject, html_body):
    if not RESEND_API_KEY:
        print("[skip] RESEND_API_KEY no seteada — omitiendo envío de email.")
        return False
    try:
        r = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "from": EMAIL_FROM,
                "to": [EMAIL_TO],
                "subject": subject,
                "html": html_body,
            },
            timeout=30,
        )
        if r.status_code in (200, 202):
            print(f"✓ Email enviado a {EMAIL_TO}")
            return True
        print(f"[email error] {r.status_code}: {r.text[:300]}")
        return False
    except Exception as e:
        print(f"[email exception] {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(f"  Weekly Auditor — {datetime.now(CHILE_TZ).strftime('%Y-%m-%d %H:%M Chile')}")
    print("=" * 60)

    yt, analytics, channel_id = get_services()
    start, end = previous_week_range()
    print(f"\nRango auditado: {start.strftime('%Y-%m-%d')} a {end.strftime('%Y-%m-%d')} (Chile)")

    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)

    uploads = get_uploads_playlist(yt, channel_id)
    video_ids = get_videos_in_range(yt, uploads, start_utc, end_utc)
    print(f"Videos en rango: {len(video_ids)}")

    videos = get_video_details(yt, video_ids)
    print(f"Videos públicos válidos: {len(videos)}")

    # Enriquecer con metrics
    today = datetime.now()
    end_str = today.strftime("%Y-%m-%d")
    for v in videos:
        v["published_at"] = datetime.fromisoformat(v["published_at"].replace("Z", "+00:00")).replace(tzinfo=None)
        m = get_video_metrics(analytics, channel_id, v["video_id"], v["published_at"].strftime("%Y-%m-%d"), end_str)
        v.update(m)
        # retention_30s desde la curva
        curve = m.get("_retention_curve") or []
        if curve and v.get("duration_sec", 0) > 0:
            target_ratio = min(30 / v["duration_sec"], 1.0)
            closest = min(curve, key=lambda r: abs(r[0] - target_ratio))
            v["_retention_30s"] = closest[1] * 100

    # Bulk fetch CTR / impressions (no se puede filtrar por video=X individualmente).
    # Pedimos por dimensión video, ventana = primer published_at hasta hoy.
    if videos:
        earliest_pub = min(v["published_at"] for v in videos).strftime("%Y-%m-%d")
        ctr_map = fetch_impressions_bulk(analytics, channel_id, earliest_pub, end_str)
        print(f"Bulk impressions/CTR fetched para {len(ctr_map)} videos del canal")
        for v in videos:
            data = ctr_map.get(v["video_id"], {})
            if data:
                v["impressions"] = data["impressions"]
                v["impressionsClickThroughRate"] = data["impressionsClickThroughRate"]

    # Baseline desde CSV
    df_all = pd.read_csv("videos_categorized.csv")
    df_all["published_at"] = pd.to_datetime(df_all["published_at"], utc=True, errors="coerce").dt.tz_localize(None)
    df_all = df_all.dropna(subset=["published_at"])
    if "averageViewDuration" not in df_all.columns:
        df_all["averageViewDuration"] = df_all.get("avg_view_duration_sec", 0)
    if "averageViewPercentage" not in df_all.columns:
        df_all["averageViewPercentage"] = df_all.get("avg_view_percentage", 0)
    df_all["impressionsClickThroughRate"] = df_all.get("ctr", 0)

    baseline = build_baseline(df_all, today, exclude_video_ids=[v["video_id"] for v in videos])
    print(f"Baseline cohort: {len(baseline)} videos (últimos {BASELINE_DAYS}d, excluyendo <{EXCLUDE_RECENT_DAYS}d)")

    # Score cada video
    items = []
    for v in videos:
        # Llenar campos categorizados desde CSV si existe
        match = df_all[df_all["video_id"] == v["video_id"]]
        if len(match):
            v["game_name"] = match.iloc[0].get("game_name", "")
            v["game_genre"] = match.iloc[0].get("game_genre", "")
            v["video_format"] = match.iloc[0].get("video_format", "")
        peers, peer_label = find_peers(v, baseline)
        score = score_video(v, peers, today)
        score["peer_label"] = peer_label
        good, bad, suggestions = analyze_video(v, score)
        items.append({
            "video": v,
            "score": score,
            "good": good,
            "bad": bad,
            "suggestions": suggestions,
        })

    saturation = detect_saturation(df_all, videos, today)
    aggregates = compute_aggregates(videos, [it["score"] for it in items], baseline)
    recommendations = build_recommendations(aggregates, items, saturation)

    # Render
    html = render_html_email(start, end, aggregates, items, saturation, recommendations)
    md = render_markdown(start, end, aggregates, items, saturation, recommendations)

    # Save MD
    Path("weekly_audits").mkdir(exist_ok=True)
    md_file = Path("weekly_audits") / f"audit_{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}.md"
    md_file.write_text(md, encoding="utf-8")
    print(f"✓ Guardado: {md_file}")

    # Update strategic_data.json
    try:
        with open("strategic_data.json") as f:
            d = json.load(f)
        # Slim version (sin curva de retención completa, etc.)
        d["weekly_audit"] = {
            "week_start": start.strftime("%Y-%m-%d"),
            "week_end": end.strftime("%Y-%m-%d"),
            "generated_at": datetime.now(CHILE_TZ).isoformat(),
            "aggregates": aggregates,
            "videos": [
                {
                    "video_id": it["video"]["video_id"],
                    "title": it["video"]["title"],
                    "thumbnail": it["video"].get("thumbnail", ""),
                    "published_at": it["video"]["published_at"].strftime("%Y-%m-%d"),
                    "days_old": it["video"]["days_old"],
                    "views": it["video"]["views"],
                    "score": it["score"],
                    "good": it["good"],
                    "bad": it["bad"],
                    "suggestions": it["suggestions"],
                }
                for it in items
            ],
            "saturation": saturation,
            "recommendations": recommendations,
        }
        with open("strategic_data.json", "w") as f:
            json.dump(d, f, ensure_ascii=False)
        print("✓ strategic_data.json actualizado con weekly_audit")
    except Exception as e:
        print(f"[warn] no pude actualizar strategic_data.json: {e}")

    # Email
    subject = f"[Joy Of Gaming] Auditoría {start.strftime('%d %b')} – {end.strftime('%d %b')}"
    if aggregates.get("video_count", 0) == 0:
        subject = f"[Joy Of Gaming] Semana sin publicación ({start.strftime('%d %b')} – {end.strftime('%d %b')})"
    send_email(subject, html)

    print("\n✓ Auditoría completa.")


if __name__ == "__main__":
    main()
