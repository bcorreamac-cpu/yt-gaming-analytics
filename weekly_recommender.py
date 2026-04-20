#!/usr/bin/env python3
"""
Weekly Recommender — Joy Of Gaming
Se ejecuta cada domingo a las 16:00 Chile (20:00 UTC).
Genera automáticamente los 5 videos de la semana siguiente.

Cron:
0 20 * * 0 cd /Users/benjamin/Desktop/YT\ Gaming\ CLI && source yt-analytics/bin/activate && python weekly_recommender.py >> weekly_log.txt 2>&1
"""

import os
import sys
import json
import pickle
import subprocess
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

CHILE_TZ_OFFSET = -4  # UTC-4 (summer time)
DAYS_OF_WEEK = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
PUBLISH_DAYS = ["Lunes", "Martes", "Miércoles", "Viernes", "Domingo"]

# Lista de juegos disponibles: OWNED + PS Plus Deluxe
OWNED_GAMES = [
    "Hellblade 2", "Forza Motorsport", "Riders Republic", "GTA V",
    "F1 2024", "Indiana Jones and the Great Circle", "Star Wars Outlaws",
    "Far Cry 6", "Tomb Raider", "Call of Duty: Modern Warfare 3",
    "Steep", "WRC", "Spider-Man 2", "RIDE 6", "Uncharted 4",
    "Watch Dogs 2", "MotoGP 25", "Cyberpunk 2077", "Venom",
]

PS_PLUS_GAMES = [
    "Alan Wake 2", "Baldur's Gate 3", "Black Myth: Wukong", "DOOM: The Dark Ages",
    "Dying Light 2", "Kingdom Come: Deliverance II", "Stellar Blade",
    "Hot Wheels Unleashed 2", "Monster Energy Supercross 6", "Dakar Desert Rally",
    "Warhammer 40K: Space Marine 2", "The Callisto Protocol", "Trek to Yomi",
    "Mafia: The Old Country", "Clair Obscur: Expedition 33", "Death Stranding 2",
    "Indiana Jones and the Great Circle", "Borderlands 4",
]

ALL_AVAILABLE = list(set(OWNED_GAMES + PS_PLUS_GAMES))


def get_youtube_service():
    """Autentica con YouTube API."""
    with open("token.pickle", "rb") as f:
        creds = pickle.load(f)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)


def get_channel_performance():
    """Analiza performance histórica del canal por juego/género."""
    df = pd.read_csv("videos_categorized.csv")
    df["engagement_rate"] = np.where(
        df["views"] > 0,
        (df["likes"] + df["comments"]) / df["views"] * 100,
        0,
    )

    # Performance por juego
    by_game = df.groupby("game_name").agg(
        n=("video_id", "count"),
        avg_views=("views", "mean"),
        avg_avd=("avg_view_duration_sec", "mean"),
        avg_eng=("engagement_rate", "mean"),
    ).reset_index()

    # Performance por género
    by_genre = df.groupby("game_genre").agg(
        avg_views=("views", "mean"),
        n=("video_id", "count"),
    ).reset_index()

    return {
        "by_game": by_game.to_dict("records"),
        "by_genre": by_genre.to_dict("records"),
        "overall_avg_views": float(df["views"].mean()),
    }


def research_game_on_youtube(youtube, game_name):
    """Busca los top videos del juego en YouTube."""
    try:
        resp = youtube.search().list(
            part="snippet",
            q=f"{game_name} PS5 4K gameplay",
            type="video",
            order="viewCount",
            maxResults=3,
            videoDuration="medium",
        ).execute()

        videos = []
        for item in resp.get("items", []):
            vid_id = item["id"]["videoId"]
            stats = youtube.videos().list(part="statistics", id=vid_id).execute()
            if stats.get("items"):
                views = int(stats["items"][0]["statistics"].get("viewCount", 0))
                videos.append({
                    "title": item["snippet"]["title"][:80],
                    "views": views,
                    "channel": item["snippet"]["channelTitle"],
                    "url": f"https://youtube.com/watch?v={vid_id}",
                })
        time.sleep(0.3)
        return sorted(videos, key=lambda x: x["views"], reverse=True)
    except Exception as e:
        print(f"  Error researching {game_name}: {e}")
        return []


def score_game(game, channel_perf, yt_research):
    """
    Calcula score de un juego para recomendación.
    Mezcla equilibrada: 33% viral potential + 33% channel history + 33% freshness
    """
    # 1. Viral potential (views de top videos en otros canales)
    viral_score = 0
    if yt_research:
        top_views = yt_research[0]["views"] if yt_research else 0
        viral_score = min(top_views / 10_000_000 * 100, 100)  # normalizado

    # 2. Channel history (cómo performa en JOG)
    history_score = 50  # default neutral
    matching_games = [g for g in channel_perf["by_game"] if game.lower() in g["game_name"].lower()]
    if matching_games:
        avg = matching_games[0]["avg_views"]
        history_score = min(avg / channel_perf["overall_avg_views"] * 50, 100)

    # 3. Freshness (juegos lanzados recientemente o AAA conocidos)
    freshness_score = 50
    new_games = ["DOOM: The Dark Ages", "Clair Obscur: Expedition 33", "Death Stranding 2",
                 "Borderlands 4", "Mafia: The Old Country", "Kingdom Come: Deliverance II",
                 "Stellar Blade", "Black Myth: Wukong", "Indiana Jones and the Great Circle"]
    if game in new_games:
        freshness_score = 90

    total = viral_score * 0.33 + history_score * 0.33 + freshness_score * 0.34
    return round(total, 1), {"viral": viral_score, "history": history_score, "fresh": freshness_score}


def generate_weekly_recommendations():
    """Genera los 5 videos de la próxima semana."""
    print("=" * 60)
    print(f"  Weekly Recommender — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # 1. Cargar performance del canal
    print("\n[1/5] Analizando performance del canal...")
    channel_perf = get_channel_performance()

    # 2. Research de YouTube para cada juego disponible
    print("\n[2/5] Investigando top videos en YouTube...")
    youtube = get_youtube_service()
    research_data = {}
    for game in ALL_AVAILABLE:
        print(f"  -> {game}")
        research_data[game] = research_game_on_youtube(youtube, game)

    # 3. Score cada juego
    print("\n[3/5] Scoring juegos...")
    scored = []
    for game in ALL_AVAILABLE:
        score, breakdown = score_game(game, channel_perf, research_data.get(game, []))
        scored.append({
            "game": game,
            "score": score,
            "breakdown": breakdown,
            "top_video": research_data[game][0] if research_data.get(game) else None,
            "in_owned": game in OWNED_GAMES,
            "in_psplus": game in PS_PLUS_GAMES,
        })
    scored.sort(key=lambda x: x["score"], reverse=True)

    # 4. Seleccionar 5 con reglas: no repetir juego, variedad de géneros
    print("\n[4/5] Seleccionando 5 videos con variedad de géneros...")
    game_genres = {
        "Spider-Man 2": "action_adventure", "RIDE 6": "racing",
        "Star Wars Outlaws": "action_adventure", "Riders Republic": "extreme_sports",
        "Hellblade 2": "action_adventure", "Black Myth: Wukong": "action_adventure",
        "Alan Wake 2": "horror", "Dying Light 2": "action_adventure",
        "Kingdom Come: Deliverance II": "rpg", "Death Stranding 2": "action_adventure",
        "DOOM: The Dark Ages": "shooter", "Stellar Blade": "action_adventure",
        "Hot Wheels Unleashed 2": "racing", "Monster Energy Supercross 6": "racing",
        "Dakar Desert Rally": "racing", "Warhammer 40K: Space Marine 2": "shooter",
        "The Callisto Protocol": "horror", "Trek to Yomi": "action_adventure",
        "Mafia: The Old Country": "action_adventure", "Clair Obscur: Expedition 33": "rpg",
        "Indiana Jones and the Great Circle": "action_adventure", "Borderlands 4": "shooter",
        "Baldur's Gate 3": "rpg", "Forza Motorsport": "racing", "GTA V": "action_adventure",
        "F1 2024": "racing", "Far Cry 6": "shooter",
        "Tomb Raider": "action_adventure", "Call of Duty: Modern Warfare 3": "shooter",
        "Steep": "extreme_sports", "WRC": "racing", "Uncharted 4": "action_adventure",
        "Watch Dogs 2": "action_adventure", "MotoGP 25": "racing",
        "Cyberpunk 2077": "rpg", "Venom": "action_adventure",
    }

    selected = []
    used_genres = {}
    for candidate in scored:
        if len(selected) >= 5:
            break
        game = candidate["game"]
        genre = game_genres.get(game, "other")
        if used_genres.get(genre, 0) >= 2:  # Max 2 del mismo género
            continue
        if game in [s["game"] for s in selected]:
            continue
        selected.append({**candidate, "genre": genre})
        used_genres[genre] = used_genres.get(genre, 0) + 1

    # 5. Construir calendario con formato completo
    print("\n[5/5] Construyendo calendario...")
    # Fechas: próxima semana (lunes a domingo)
    today = datetime.now()
    days_until_monday = (7 - today.weekday()) % 7 or 7
    next_monday = today + timedelta(days=days_until_monday)

    date_map = {}
    for i, day in enumerate(PUBLISH_DAYS):
        offset = DAYS_OF_WEEK.index(day)
        date_map[day] = (next_monday + timedelta(days=offset)).strftime("%Y-%m-%d")

    calendar_entries = []
    for idx, sel in enumerate(selected):
        day = PUBLISH_DAYS[idx]
        game = sel["game"]
        top = sel["top_video"]

        calendar_entries.append({
            "date": date_map[day],
            "day": day,
            "game": game,
            "genre": sel["genre"],
            "format": "Showcase Visual" if sel["genre"] != "racing" else "Primera Persona POV",
            "mission": f"Generar misión específica para {game} — ver audit manual",
            "game_progress": "A definir",
            "details": f"{game} tiene alto potencial. Top video de referencia: {top['views']:,} views en {top['channel']}" if top else f"Explorar {game}.",
            "title": f"(PS5) {game} - INSANE Gameplay | Ultra Realistic Graphics [4K HDR 60FPS]",
            "ref_video": f"https://www.youtube.com/results?search_query={game.replace(' ','+')}+PS5+4K",
            "ref_channel": top["channel"] if top else "Sin data",
            "predicted_views": int(1000 + sel["score"] * 50),
            "predicted_low": int(500 + sel["score"] * 20),
            "predicted_high": int(2000 + sel["score"] * 150),
            "miniatura": f"MODO FOTO PS5: Captura icónica de {game}. Composición épica sin HUD. Ratio 16:9.",
            "ai_prompt": f"Photorealistic screenshot from {game} videogame. Epic composition. 4K cinematic quality. No text, no UI.",
            "source": "OWNED" if sel["in_owned"] else "PS+",
            "auto_score": sel["score"],
        })

    # 6. Actualizar strategic_data.json
    with open("strategic_data.json") as f:
        data = json.load(f)
    data["calendar"] = calendar_entries
    with open("strategic_data.json", "w") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"\n✓ Calendario generado con {len(calendar_entries)} videos:")
    for c in calendar_entries:
        print(f"  {c['date']} {c['day']:10s} | {c['game']:30s} | Score: {c['auto_score']}")

    # 7. Inyectar en strategic_dashboard.html y push
    print("\nActualizando dashboard...")
    subprocess.run(["python3", "rebuild_dashboard.py"])
    # Note: git commit and push are handled by GitHub Actions workflow
    # When run locally, uncomment the lines below:
    # subprocess.run(["git", "add", "strategic_dashboard.html", "strategic_data.json"])
    # subprocess.run(["git", "commit", "-m", f"Auto: Weekly calendar {next_monday.strftime('%Y-%m-%d')}"])
    # subprocess.run(["git", "push"])

    print(f"\n✓ Files updated: strategic_data.json + strategic_dashboard.html")
    print("=" * 60)


if __name__ == "__main__":
    generate_weekly_recommendations()
