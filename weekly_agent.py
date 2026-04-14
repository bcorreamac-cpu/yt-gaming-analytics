#!/usr/bin/env python3
"""
Agente Semanal — Joy Of Gaming
Analiza el rendimiento de los videos de la semana anterior y genera
recomendaciones para la semana siguiente.

Uso: python weekly_agent.py
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

TOKEN_FILE = "token.pickle"
CALENDAR_FILE = "strategic_data.json"
REPORT_DIR = "weekly_reports"

def get_youtube_service():
    """Conecta con YouTube Data API usando token existente."""
    if not os.path.exists(TOKEN_FILE):
        print("ERROR: No hay token. Ejecuta data_collector.py primero.")
        return None
    with open(TOKEN_FILE, "rb") as f:
        creds = pickle.load(f)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)


def get_recent_videos(youtube, days=7):
    """Obtiene los videos publicados en los últimos N días."""
    after = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")

    # Get channel uploads playlist
    ch = youtube.channels().list(part="contentDetails", mine=True).execute()
    uploads = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    videos = []
    page_token = None
    while True:
        resp = youtube.playlistItems().list(
            part="contentDetails,snippet",
            playlistId=uploads,
            maxResults=50,
            pageToken=page_token,
        ).execute()

        for item in resp.get("items", []):
            pub = item["snippet"]["publishedAt"]
            if pub >= after:
                videos.append(item["contentDetails"]["videoId"])
            else:
                # Videos are in reverse chronological order
                break

        page_token = resp.get("nextPageToken")
        if not page_token or (resp.get("items") and resp["items"][-1]["snippet"]["publishedAt"] < after):
            break

    if not videos:
        return pd.DataFrame()

    # Get details
    details = []
    for i in range(0, len(videos), 50):
        batch = ",".join(videos[i:i+50])
        resp = youtube.videos().list(part="snippet,statistics,contentDetails", id=batch).execute()
        for v in resp.get("items", []):
            stats = v.get("statistics", {})
            details.append({
                "video_id": v["id"],
                "title": v["snippet"]["title"],
                "published_at": v["snippet"]["publishedAt"],
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0)),
            })

    df = pd.DataFrame(details)
    if not df.empty:
        df["engagement_rate"] = np.where(
            df["views"] > 0,
            (df["likes"] + df["comments"]) / df["views"] * 100,
            0,
        )
    return df


def analyze_week(df):
    """Analiza los resultados de la semana."""
    if df.empty:
        return {"status": "sin_datos", "message": "No se publicaron videos esta semana."}

    analysis = {
        "status": "ok",
        "num_videos": len(df),
        "total_views": int(df["views"].sum()),
        "avg_views": round(df["views"].mean(), 0),
        "best_video": df.loc[df["views"].idxmax()].to_dict(),
        "worst_video": df.loc[df["views"].idxmin()].to_dict(),
        "avg_engagement": round(df["engagement_rate"].mean(), 3),
        "total_likes": int(df["likes"].sum()),
        "videos": df.sort_values("views", ascending=False).to_dict("records"),
    }
    return analysis


def generate_recommendations(analysis, historical_data):
    """Genera recomendaciones basadas en el análisis semanal."""
    recs = []

    if analysis["status"] == "sin_datos":
        recs.append({
            "priority": "CRÍTICA",
            "action": "Publicar contenido esta semana",
            "detail": "No se publicaron videos. La consistencia es clave para el algoritmo de YouTube.",
        })
        return recs

    # Compare with historical averages
    hist_avg = historical_data.get("overview", {}).get("avg_views_per_video", 0)
    week_avg = analysis["avg_views"]

    if week_avg > hist_avg * 1.5:
        recs.append({
            "priority": "POSITIVO",
            "action": "El contenido de esta semana superó el promedio histórico",
            "detail": f"Avg views: {week_avg:,.0f} vs histórico: {hist_avg:,.0f}. "
                      f"Analizar qué hizo diferente al mejor video: {analysis['best_video'].get('title', 'N/A')}",
        })
    elif week_avg < hist_avg * 0.5:
        recs.append({
            "priority": "ALERTA",
            "action": "Views por debajo del promedio",
            "detail": f"Avg views: {week_avg:,.0f} vs histórico: {hist_avg:,.0f}. "
                      "Considerar cambiar el tipo de contenido o mejorar thumbnails y títulos.",
        })

    # Check engagement
    if analysis["avg_engagement"] > 2.0:
        recs.append({
            "priority": "POSITIVO",
            "action": "Engagement alto — la audiencia está conectada",
            "detail": f"Engagement: {analysis['avg_engagement']:.2f}%. Mantener este tipo de contenido.",
        })
    elif analysis["avg_engagement"] < 1.0:
        recs.append({
            "priority": "ALERTA",
            "action": "Engagement bajo",
            "detail": "Agregar call-to-action para likes y comentarios. Hacer preguntas en la descripción.",
        })

    # Best video analysis
    best = analysis["best_video"]
    recs.append({
        "priority": "INSIGHT",
        "action": f"Mejor video: {best.get('title', 'N/A')[:60]}",
        "detail": f"Views: {best.get('views', 0):,}. Crear más contenido similar a este formato.",
    })

    # Volume check
    if analysis["num_videos"] < 4:
        recs.append({
            "priority": "ACCIÓN",
            "action": f"Solo {analysis['num_videos']} videos publicados",
            "detail": "Meta: 5 videos/semana. Aumentar frecuencia de publicación.",
        })

    return recs


def save_report(analysis, recommendations, week_label):
    """Guarda el reporte semanal."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    report = {
        "week": week_label,
        "generated_at": datetime.now().isoformat(),
        "analysis": analysis,
        "recommendations": recommendations,
    }

    filename = os.path.join(REPORT_DIR, f"week_{week_label}.json")
    with open(filename, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    return filename


def main():
    print("=" * 60)
    print("  Agente Semanal — Joy Of Gaming")
    print("=" * 60)

    # Load historical data
    with open(CALENDAR_FILE) as f:
        hist = json.load(f)

    # Connect to YouTube
    youtube = get_youtube_service()
    if not youtube:
        return

    # Get last 7 days of videos
    print("\nObteniendo videos de los últimos 7 días...")
    df = get_recent_videos(youtube, days=7)

    # Analyze
    print("Analizando rendimiento...")
    analysis = analyze_week(df)

    # Generate recommendations
    print("Generando recomendaciones...")
    recs = generate_recommendations(analysis, hist)

    # Save report
    week_label = datetime.now().strftime("%Y-%m-%d")
    filename = save_report(analysis, recs, week_label)

    # Print results
    print(f"\n{'='*60}")
    print(f"  REPORTE SEMANAL — {week_label}")
    print(f"{'='*60}")

    if analysis["status"] == "sin_datos":
        print("\n  No se publicaron videos esta semana.")
    else:
        print(f"\n  Videos publicados: {analysis['num_videos']}")
        print(f"  Views totales:     {analysis['total_views']:,}")
        print(f"  Views promedio:    {analysis['avg_views']:,.0f}")
        print(f"  Engagement:        {analysis['avg_engagement']:.2f}%")
        print(f"\n  Mejor video: {analysis['best_video'].get('title', 'N/A')[:60]}")
        print(f"               {analysis['best_video'].get('views', 0):,} views")

    print(f"\n  RECOMENDACIONES:")
    for r in recs:
        print(f"  [{r['priority']}] {r['action']}")
        print(f"    → {r['detail']}")

    print(f"\n  Reporte guardado: {filename}")
    print("=" * 60)


if __name__ == "__main__":
    main()
