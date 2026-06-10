#!/usr/bin/env python3
"""Check if YouTube has new videos not yet in videos_raw_data.csv.

Cheap operation (~1 API call) used by quick-refresh.yml workflow to
avoid running the full pipeline every 30 min when nothing changed.

Exits 0 always. Prints "yes" to stdout if new videos detected,
"no" otherwise. The workflow conditionally runs the full pipeline based on this.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from googleapiclient.discovery import build

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "videos_raw_data.csv"
TOKEN = ROOT / "token.pickle"
CHANNEL_ID = "UCCIR3AdAbybtkNgINS9UyNA"  # Joy Of Gaming
UPLOADS_PLAYLIST = "UU" + CHANNEL_ID[2:]
CHECK_LATEST_N = 10  # only look at the most recent N uploads


def get_known_video_ids() -> set[str]:
    if not CSV.exists():
        return set()
    try:
        df = pd.read_csv(CSV, usecols=["video_id"])
        return set(df["video_id"].dropna().astype(str).tolist())
    except Exception as e:
        print(f"[warn] no pude leer CSV: {e}", file=sys.stderr)
        return set()


def get_latest_video_ids(youtube) -> list[str]:
    resp = youtube.playlistItems().list(
        part="contentDetails",
        playlistId=UPLOADS_PLAYLIST,
        maxResults=CHECK_LATEST_N,
    ).execute()
    return [item["contentDetails"]["videoId"] for item in resp.get("items", [])]


def main() -> int:
    if not TOKEN.exists():
        print("no")  # sin token no podemos chequear; fallback seguro
        print("[error] token.pickle no encontrado", file=sys.stderr)
        return 0

    with open(TOKEN, "rb") as f:
        creds = pickle.load(f)
    youtube = build("youtube", "v3", credentials=creds)

    known = get_known_video_ids()
    latest = get_latest_video_ids(youtube)

    new_ids = [vid for vid in latest if vid not in known]

    if new_ids:
        print("yes")
        print(f"[info] {len(new_ids)} videos nuevos detectados: {new_ids}", file=sys.stderr)
    else:
        print("no")
        print(f"[info] sin novedades. CSV tiene {len(known)} videos.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
