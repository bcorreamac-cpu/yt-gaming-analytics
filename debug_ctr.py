"""Debug CTR fetch for a sample of videos.

Usage:
    python debug_ctr.py

Prints the raw Analytics API response for impressions+CTR for 5 recent videos.
If errors occur, prints the FULL HttpError so we can see why CTR is 0.
"""
import pickle
import json
import pandas as pd
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

with open("token.pickle", "rb") as f:
    creds = pickle.load(f)

if creds.expired and creds.refresh_token:
    from google.auth.transport.requests import Request
    creds.refresh(Request())
    with open("token.pickle", "wb") as f:
        pickle.dump(creds, f)
    print("Token refreshed.")

yt = build("youtube", "v3", credentials=creds)
analytics = build("youtubeAnalytics", "v2", credentials=creds)

ch = yt.channels().list(part="id", mine=True).execute()
channel_id = ch["items"][0]["id"]
print(f"Channel ID: {channel_id}\n")

df = pd.read_csv("videos_categorized.csv")
df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
recent = df.sort_values("published_at", ascending=False).head(5)

end_date = datetime.now().strftime("%Y-%m-%d")

print("Probando 5 videos más recientes:\n")
for _, row in recent.iterrows():
    vid = row["video_id"]
    pub = str(row["published_at"])[:10]
    title = row["title"][:60]
    print(f"--- {vid}  {pub}  {title} ---")

    # Query 1: impressions + CTR (esto es lo que está fallando silencioso)
    try:
        resp = analytics.reports().query(
            ids=f"channel=={channel_id}",
            startDate=pub,
            endDate=end_date,
            metrics="impressions,impressionsClickThroughRate",
            filters=f"video=={vid}",
        ).execute()
        print(f"  Response: {json.dumps(resp, indent=2)[:500]}")
    except HttpError as e:
        print(f"  HttpError {e.resp.status}: {e.content.decode() if e.content else 'no content'}")
    except Exception as e:
        print(f"  Other error: {type(e).__name__}: {e}")
    print()
