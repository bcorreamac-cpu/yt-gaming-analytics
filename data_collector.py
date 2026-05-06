#!/usr/bin/env python3
"""
YouTube Data & Analytics Collector
Extrae datos completos de un canal de YouTube usando Data API v3 y Analytics API.
"""

import os
import sys
import glob
import json
import time
import random
import pickle
import isodate
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
SCOPES = [
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]
TOKEN_FILE = "token.pickle"
OUTPUT_CSV = "videos_raw_data.csv"
TOP_N_RETENTION = 200
MAX_RESULTS_PER_PAGE = 50

# ---------------------------------------------------------------------------
# Autenticación OAuth 2.0
# ---------------------------------------------------------------------------

def find_client_secret():
    """Busca el archivo client_secret*.json en el directorio actual."""
    matches = glob.glob("client_secret*.json")
    if not matches:
        print("ERROR: No se encontró ningún archivo client_secret*.json")
        print("Descarga tu archivo de credenciales OAuth desde Google Cloud Console")
        print("y colócalo en este directorio.")
        sys.exit(1)
    return matches[0]


def authenticate():
    """Autentica con OAuth 2.0 y devuelve credenciales."""
    creds = None

    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Renovando token de acceso...")
            creds.refresh(Request())
        else:
            client_secret = find_client_secret()
            print(f"Usando credenciales: {client_secret}")
            flow = InstalledAppFlow.from_client_secrets_file(client_secret, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "wb") as f:
            pickle.dump(creds, f)
        print("Autenticación exitosa. Token guardado.")

    return creds


# ---------------------------------------------------------------------------
# Rate limiting con exponential backoff
# ---------------------------------------------------------------------------

def api_call_with_backoff(request_func, max_retries=5):
    """Ejecuta una llamada a la API con exponential backoff."""
    for attempt in range(max_retries):
        try:
            return request_func()
        except HttpError as e:
            if e.resp.status in (403, 429, 500, 503):
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"\n  Rate limit/error (HTTP {e.resp.status}). "
                      f"Reintentando en {wait:.1f}s (intento {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise Exception(f"Máximo de reintentos ({max_retries}) alcanzado.")


# ---------------------------------------------------------------------------
# Obtener ID del canal autenticado
# ---------------------------------------------------------------------------

def get_my_channel_info(youtube):
    """Obtiene el ID del canal y el playlist de uploads del usuario autenticado."""
    resp = api_call_with_backoff(
        lambda: youtube.channels().list(
            part="id,snippet,contentDetails", mine=True
        ).execute()
    )
    if not resp.get("items"):
        print("ERROR: No se encontró un canal asociado a esta cuenta.")
        sys.exit(1)
    channel = resp["items"][0]
    channel_id = channel["id"]
    uploads_playlist = channel["contentDetails"]["relatedPlaylists"]["uploads"]
    print(f"Canal: {channel['snippet']['title']} ({channel_id})")
    print(f"Playlist de uploads: {uploads_playlist}")
    return channel_id, uploads_playlist


# ---------------------------------------------------------------------------
# Obtener TODOS los videos con paginación vía playlistItems
# ---------------------------------------------------------------------------

def get_all_video_ids(youtube, uploads_playlist_id):
    """Obtiene todos los IDs de videos usando playlistItems.list (sin límite de 500)."""
    video_ids = []
    page_token = None
    print("\nObteniendo lista completa de videos del canal...")

    while True:
        resp = api_call_with_backoff(
            lambda pt=page_token: youtube.playlistItems().list(
                part="contentDetails",
                playlistId=uploads_playlist_id,
                maxResults=MAX_RESULTS_PER_PAGE,
                pageToken=pt,
            ).execute()
        )
        for item in resp.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])

        page_token = resp.get("nextPageToken")
        total_expected = resp.get("pageInfo", {}).get("totalResults", "?")
        print(f"  Videos encontrados: {len(video_ids)} / {total_expected}", end="\r")

        if not page_token:
            break

    print(f"\nTotal de videos encontrados: {len(video_ids)}")
    return video_ids


# ---------------------------------------------------------------------------
# Obtener detalles y métricas básicas de videos (en lotes de 50)
# ---------------------------------------------------------------------------

def parse_duration(iso_duration):
    """Convierte duración ISO 8601 a segundos."""
    try:
        return int(isodate.parse_duration(iso_duration).total_seconds())
    except Exception:
        return 0


def get_video_details(youtube, video_ids):
    """Obtiene detalles completos de videos en lotes de 50."""
    all_details = []

    for i in tqdm(range(0, len(video_ids), MAX_RESULTS_PER_PAGE),
                  desc="Obteniendo detalles de videos"):
        batch_ids = video_ids[i:i + MAX_RESULTS_PER_PAGE]
        ids_str = ",".join(batch_ids)

        resp = api_call_with_backoff(
            lambda ids=ids_str: youtube.videos().list(
                part="snippet,contentDetails,statistics,status",
                id=ids,
            ).execute()
        )

        for item in resp.get("items", []):
            snippet = item["snippet"]
            stats = item.get("statistics", {})
            content = item["contentDetails"]
            status = item.get("status", {})

            title = snippet.get("title", "")
            duration_sec = parse_duration(content.get("duration", "PT0S"))

            # Filtrar Shorts: duración < 60s o título con #Shorts/#shorts
            if duration_sec < 60:
                continue
            if re.search(r"#[Ss]horts", title):
                continue

            # Filtrar videos no públicos: privados, unlisted, deleted, fallidos
            if status.get("privacyStatus") != "public":
                continue
            if status.get("uploadStatus") != "processed":
                continue

            thumbnails = snippet.get("thumbnails", {})
            thumb_url = (
                thumbnails.get("maxres", {}).get("url")
                or thumbnails.get("high", {}).get("url")
                or thumbnails.get("default", {}).get("url", "")
            )

            all_details.append({
                "video_id": item["id"],
                "title": title,
                "description": snippet.get("description", ""),
                "tags": "|".join(snippet.get("tags", [])),
                "published_at": snippet.get("publishedAt", ""),
                "duration_seconds": duration_sec,
                "thumbnail_url": thumb_url,
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0)),
                # shares no está disponible en Data API v3 statistics
                "shares": 0,
            })

    return all_details


# ---------------------------------------------------------------------------
# Métricas avanzadas vía YouTube Analytics API
# ---------------------------------------------------------------------------

def get_analytics_metrics(analytics, channel_id, video_id, start_date, end_date):
    """Obtiene métricas avanzadas de Analytics API para un video.
    Separa impressions en una segunda query con fallback."""
    result = {}

    # Query 1: métricas básicas de analytics (siempre disponibles)
    try:
        resp = api_call_with_backoff(
            lambda: analytics.reports().query(
                ids=f"channel=={channel_id}",
                startDate=start_date,
                endDate=end_date,
                metrics=(
                    "views,estimatedMinutesWatched,averageViewDuration,"
                    "averageViewPercentage,subscribersGained,shares"
                ),
                filters=f"video=={video_id}",
            ).execute()
        )
        if resp.get("rows") and len(resp["rows"]) > 0:
            row = resp["rows"][0]
            headers = [col["name"] for col in resp["columnHeaders"]]
            result.update(dict(zip(headers, row)))
    except HttpError as e:
        if e.resp.status != 403:
            print(f"\n  Analytics error para {video_id}: {e}")

    # Query 2: impressions y CTR (no disponible para todos los videos)
    try:
        resp2 = api_call_with_backoff(
            lambda: analytics.reports().query(
                ids=f"channel=={channel_id}",
                startDate=start_date,
                endDate=end_date,
                metrics="impressions,impressionsClickThroughRate",
                filters=f"video=={video_id}",
            ).execute()
        )
        if resp2.get("rows") and len(resp2["rows"]) > 0:
            row2 = resp2["rows"][0]
            headers2 = [col["name"] for col in resp2["columnHeaders"]]
            result.update(dict(zip(headers2, row2)))
    except HttpError as e:
        # Loggear primer error para diagnóstico (antes era silent)
        if not hasattr(get_analytics_metrics, "_ctr_err_logged"):
            print(f"\n  [CTR debug] HttpError en impressions/CTR para {video_id}: "
                  f"status={e.resp.status}, content={e.content[:200] if e.content else 'empty'}")
            get_analytics_metrics._ctr_err_logged = True

    return result


def enrich_with_analytics(analytics, channel_id, videos):
    """Enriquece los datos de videos con métricas de Analytics API."""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = "2000-01-01"

    print("\nObteniendo métricas avanzadas de Analytics API...")
    for video in tqdm(videos, desc="Analytics por video"):
        if video.get("published_at"):
            pub_date = video["published_at"][:10]
        else:
            pub_date = start_date

        metrics = get_analytics_metrics(
            analytics, channel_id, video["video_id"], pub_date, end_date
        )

        video["impressions"] = metrics.get("impressions", 0)
        video["ctr"] = metrics.get("impressionsClickThroughRate", 0)
        video["avg_view_duration_sec"] = metrics.get("averageViewDuration", 0)
        video["avg_view_percentage"] = metrics.get("averageViewPercentage", 0)
        video["subscribers_gained"] = metrics.get("subscribersGained", 0)
        video["estimated_minutes_watched"] = metrics.get("estimatedMinutesWatched", 0)
        # Actualizar shares desde Analytics (más preciso que Data API)
        if metrics.get("shares", 0) > 0:
            video["shares"] = metrics.get("shares", 0)

    return videos


# ---------------------------------------------------------------------------
# Fuentes de tráfico por video
# ---------------------------------------------------------------------------

def get_traffic_sources(analytics, channel_id, video_id, start_date, end_date):
    """Obtiene fuentes de tráfico para un video."""
    try:
        resp = api_call_with_backoff(
            lambda: analytics.reports().query(
                ids=f"channel=={channel_id}",
                startDate=start_date,
                endDate=end_date,
                metrics="views,estimatedMinutesWatched",
                dimensions="insightTrafficSourceType",
                filters=f"video=={video_id}",
                sort="-views",
            ).execute()
        )
        if resp.get("rows"):
            sources = {}
            for row in resp["rows"]:
                sources[row[0]] = {"views": row[1], "minutes": row[2]}
            return json.dumps(sources, ensure_ascii=False)
    except HttpError:
        pass
    return "{}"


def enrich_with_traffic_sources(analytics, channel_id, videos):
    """Agrega fuentes de tráfico a cada video."""
    end_date = datetime.now().strftime("%Y-%m-%d")

    print("\nObteniendo fuentes de tráfico por video...")
    for video in tqdm(videos, desc="Fuentes de tráfico"):
        pub_date = video.get("published_at", "2000-01-01")[:10]
        video["traffic_sources"] = get_traffic_sources(
            analytics, channel_id, video["video_id"], pub_date, end_date
        )

    return videos


# ---------------------------------------------------------------------------
# Curva de retención (top N videos)
# ---------------------------------------------------------------------------

def get_retention_data(analytics, channel_id, video_id, start_date, end_date):
    """Obtiene la curva de retención de audiencia para un video."""
    try:
        resp = api_call_with_backoff(
            lambda: analytics.reports().query(
                ids=f"channel=={channel_id}",
                startDate=start_date,
                endDate=end_date,
                metrics="audienceWatchRatio",
                dimensions="elapsedVideoTimeRatio",
                filters=f"video=={video_id}",
                sort="elapsedVideoTimeRatio",
            ).execute()
        )
        if resp.get("rows"):
            return resp["rows"]  # [[ratio_tiempo, ratio_audiencia], ...]
    except HttpError:
        pass
    return []


def calculate_retention_metrics(retention_curve, duration_seconds):
    """Calcula retención en puntos clave a partir de la curva."""
    if not retention_curve or duration_seconds <= 0:
        return {
            "retention_30s": None,
            "retention_1min": None,
            "retention_50pct": None,
            "retention_70pct": None,
        }

    # Crear mapeo de ratio de tiempo -> ratio de audiencia
    time_to_retention = {}
    for row in retention_curve:
        time_ratio = row[0]
        audience_ratio = row[1]
        time_to_retention[time_ratio] = audience_ratio

    def get_retention_at_seconds(target_seconds):
        """Obtiene retención al segundo especificado usando interpolación."""
        if duration_seconds <= 0:
            return None
        target_ratio = target_seconds / duration_seconds
        if target_ratio > 1.0:
            return None

        # Encontrar los puntos más cercanos
        ratios = sorted(time_to_retention.keys())
        if not ratios:
            return None

        # Interpolación lineal
        prev_ratio = 0
        prev_retention = 1.0
        for r in ratios:
            if r >= target_ratio:
                if r == target_ratio:
                    return round(time_to_retention[r] * 100, 2)
                # Interpolar
                frac = (target_ratio - prev_ratio) / (r - prev_ratio) if r != prev_ratio else 0
                ret = prev_retention + frac * (time_to_retention[r] - prev_retention)
                return round(ret * 100, 2)
            prev_ratio = r
            prev_retention = time_to_retention[r]

        return round(time_to_retention[ratios[-1]] * 100, 2)

    def get_retention_at_percentage(target_pct):
        """Obtiene retención al X% del video."""
        target_ratio = target_pct / 100.0
        ratios = sorted(time_to_retention.keys())
        if not ratios:
            return None

        prev_ratio = 0
        prev_retention = 1.0
        for r in ratios:
            if r >= target_ratio:
                if r == target_ratio:
                    return round(time_to_retention[r] * 100, 2)
                frac = (target_ratio - prev_ratio) / (r - prev_ratio) if r != prev_ratio else 0
                ret = prev_retention + frac * (time_to_retention[r] - prev_retention)
                return round(ret * 100, 2)
            prev_ratio = r
            prev_retention = time_to_retention[r]

        return round(time_to_retention[ratios[-1]] * 100, 2)

    return {
        "retention_30s": get_retention_at_seconds(30),
        "retention_1min": get_retention_at_seconds(60),
        "retention_50pct": get_retention_at_percentage(50),
        "retention_70pct": get_retention_at_percentage(70),
    }


def enrich_with_retention(analytics, channel_id, videos):
    """Agrega datos de retención a TODOS los videos.

    Estrategia incremental para no quemar cuota de la Analytics API:
      - Si el video ya tiene retention_curve en el CSV anterior y se publicó
        hace >90 días: reusa el cache (retención estable).
      - Si el video se publicó en los últimos 90 días o nunca tuvo curva:
        re-fetcha (retención reciente todavía evoluciona).
    """
    import os as _os

    cache: dict = {}
    csv_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "videos_raw_data.csv")
    if _os.path.exists(csv_path):
        try:
            prev = pd.read_csv(csv_path)
            for _, row in prev.iterrows():
                vid = row.get("video_id")
                if not vid or pd.isna(vid):
                    continue
                curve = row.get("retention_curve") or ""
                if isinstance(curve, str) and curve:
                    cache[vid] = {
                        "retention_30s": None if pd.isna(row.get("retention_30s")) else row.get("retention_30s"),
                        "retention_1min": None if pd.isna(row.get("retention_1min")) else row.get("retention_1min"),
                        "retention_50pct": None if pd.isna(row.get("retention_50pct")) else row.get("retention_50pct"),
                        "retention_70pct": None if pd.isna(row.get("retention_70pct")) else row.get("retention_70pct"),
                        "retention_curve": curve,
                    }
        except Exception as e:
            print(f"  [warn] no pude leer cache de retención: {e}")

    print(f"\nObteniendo curvas de retención para TODOS los videos ({len(videos)})...")
    print(f"  Cache previo: {len(cache)} videos con retención ya capturada")

    end_date = datetime.now().strftime("%Y-%m-%d")
    today = datetime.now().date()
    fetched = 0
    cached_used = 0

    for video in tqdm(videos, desc="Retención de audiencia"):
        vid = video["video_id"]
        pub_str = (video.get("published_at") or "2000-01-01")[:10]
        try:
            pub_date_dt = datetime.strptime(pub_str, "%Y-%m-%d").date()
            days_old = (today - pub_date_dt).days
        except Exception:
            days_old = 9999

        cached = cache.get(vid)
        if cached and days_old > 90:
            video["retention_30s"] = cached["retention_30s"]
            video["retention_1min"] = cached["retention_1min"]
            video["retention_50pct"] = cached["retention_50pct"]
            video["retention_70pct"] = cached["retention_70pct"]
            video["retention_curve"] = cached["retention_curve"]
            cached_used += 1
            continue

        curve = get_retention_data(
            analytics, channel_id, vid, pub_str, end_date
        )
        metrics = calculate_retention_metrics(curve, video.get("duration_seconds", 0))
        video["retention_30s"] = metrics["retention_30s"]
        video["retention_1min"] = metrics["retention_1min"]
        video["retention_50pct"] = metrics["retention_50pct"]
        video["retention_70pct"] = metrics["retention_70pct"]
        video["retention_curve"] = json.dumps(curve) if curve else ""
        fetched += 1

    print(f"  Resultado: {fetched} re-fetched | {cached_used} desde cache")
    return videos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  YouTube Data & Analytics Collector")
    print("=" * 60)

    # 1. Autenticación
    creds = authenticate()

    youtube = build("youtube", "v3", credentials=creds)
    analytics = build("youtubeAnalytics", "v2", credentials=creds)

    # 2. Obtener canal y playlist de uploads
    channel_id, uploads_playlist = get_my_channel_info(youtube)

    # 3. Obtener todos los video IDs vía playlist de uploads
    video_ids = get_all_video_ids(youtube, uploads_playlist)

    if not video_ids:
        print("No se encontraron videos en el canal.")
        sys.exit(0)

    # 4. Obtener detalles y métricas básicas (filtra Shorts automáticamente)
    videos = get_video_details(youtube, video_ids)
    shorts_filtered = len(video_ids) - len(videos)
    print(f"Videos long-form: {len(videos)} (Shorts excluidos: {shorts_filtered})")

    # 5. Métricas avanzadas vía Analytics API
    videos = enrich_with_analytics(analytics, channel_id, videos)

    # 6. Fuentes de tráfico
    videos = enrich_with_traffic_sources(analytics, channel_id, videos)

    # 7. Curvas de retención (top 200)
    videos = enrich_with_retention(analytics, channel_id, videos)

    # 8. Guardar CSV
    df = pd.DataFrame(videos)
    column_order = [
        "video_id", "title", "description", "tags", "published_at",
        "duration_seconds", "thumbnail_url",
        "views", "likes", "comments", "shares",
        "impressions", "ctr", "avg_view_duration_sec", "avg_view_percentage",
        "subscribers_gained", "estimated_minutes_watched",
        "retention_30s", "retention_1min", "retention_50pct", "retention_70pct",
        "retention_curve", "traffic_sources",
    ]
    existing_cols = [c for c in column_order if c in df.columns]
    df = df[existing_cols]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nDatos guardados en: {OUTPUT_CSV}")
    print(f"Total de videos: {len(df)}")
    print(f"Columnas: {len(df.columns)}")

    # Resumen rápido
    print("\n--- Resumen ---")
    print(f"Total views:    {df['views'].sum():,}")
    print(f"Total likes:    {df['likes'].sum():,}")
    print(f"Total comments: {df['comments'].sum():,}")
    if "estimated_minutes_watched" in df.columns:
        total_min = df["estimated_minutes_watched"].sum()
        print(f"Total minutos:  {total_min:,.0f} ({total_min / 60:,.0f} horas)")
    print("=" * 60)


if __name__ == "__main__":
    main()
