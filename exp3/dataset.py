"""Building the Experiment 3 video sample (port of analyze_sponsor_data.ipynb + censorer.py).

The paper's sample is stored in data/videos_censored.json; this module documents and
can repeat how it was built:
  1. SponsorBlock segments with category 'sponsor' lasting 20-240 s, shuffled.
  2. For each segment, YouTube Data API metadata (title, channel name, channel
     description); skip if any is missing.
  3. English transcript text whose start time falls in [start-5 s, min(end+5 s, duration)];
     skip if under 40 words.
  4. GPT-4o checks that the excerpt is a sponsorship. The original cell passed
     `prompt=` to the chat API with an unimported client and could not run as written;
     this is the evidently intended chat call.
  5. Stop at 300 videos. GPT-4o then rewrites each transcript with brand names replaced
     by generic descriptions and drops non-sponsor excerpts.
Both prompts are in dataset_prompts.yaml.
Requires: pip install -e .[collect]; YOUTUBE_API_KEY and OPENAI_API_KEY in .env.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from common import files, llm
from common.config import load_models, load_prompts


def _cfg() -> dict:
    return load_prompts("exp3", "dataset_prompts")


def _ask(prompt: str) -> str:
    """One single-turn request to the dataset model (no token cap or temperature, as originally)."""
    return llm.ask(llm.chat, load_models()["models"][_cfg()["model"]], [], prompt)[-1]["content"]


def candidate_segments(sponsor_times_csv: Path, seed: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(sponsor_times_csv)
    df = df[df["category"] == "sponsor"]
    df = df[(df["endTime"] - df["startTime"] >= 20) & (df["endTime"] - df["startTime"] < 240)]
    for i in range(10):
        df = df.sample(frac=1, random_state=None if seed is None else seed + i).reset_index(drop=True)
    return df


def video_and_channel_info(video_id: str):
    import isodate
    import requests

    key = os.environ["YOUTUBE_API_KEY"]
    v = requests.get("https://www.googleapis.com/youtube/v3/videos",
                     params={"part": "snippet,contentDetails", "id": video_id, "key": key}).json()
    if not v.get("items"):
        return None, None, None, None
    snippet, details = v["items"][0]["snippet"], v["items"][0]["contentDetails"]
    duration = isodate.parse_duration(details["duration"]).total_seconds()
    c = requests.get("https://www.googleapis.com/youtube/v3/channels",
                     params={"part": "snippet", "id": snippet["channelId"], "key": key}).json()
    if not c.get("items"):
        return snippet["title"], snippet["channelTitle"], None, duration
    return snippet["title"], snippet["channelTitle"], c["items"][0]["snippet"]["description"], duration


def transcript_segment(video_id: str, start: float, end: float) -> str | None:
    from youtube_transcript_api import YouTubeTranscriptApi
    try:
        transcript = YouTubeTranscriptApi.list_transcripts(video_id).find_transcript(["en"]).fetch()
        return " ".join(e.text for e in transcript if start <= e.start <= end)
    except Exception as e:  # noqa: BLE001 - any transcript failure means "skip this video"
        print(f"Skipping video {video_id}: {e}")
        return None


def build(sponsor_times_csv: Path, out: Path, n: int = 300, seed: int | None = None) -> Path:
    df = candidate_segments(sponsor_times_csv, seed)
    videos: list[dict] = []
    for _, row in df.iterrows():
        if len(videos) >= n:
            break
        vid = row["videoID"]
        title, channel, description, duration = video_and_channel_info(vid)
        if title is None or channel is None or description is None:
            continue
        text = transcript_segment(vid, max(row["startTime"] - 5, 0), min(row["endTime"] + 5, duration))
        if text is None or len(text.split()) < 40:
            continue
        if "yes" not in _ask(_cfg()["sponsor_check"].format(transcript=text)).lower():
            continue
        videos.append({"video_id": vid, "video_title": title, "video_channel_name": channel,
                       "video_channel_description": description, "video_sponsorship_transcript": text})
        files.write_json(out, videos)
    return out


def censor(in_path: Path, out_path: Path) -> Path:
    cfg = _cfg()
    kept = []
    for entry in files.read_json(in_path):
        original = entry.get("video_sponsorship_transcript", "")
        if not original.strip():
            continue
        result = _ask(cfg["censor"].format(transcript=original.strip())).strip()
        if result == cfg["no_sponsor_token"]:
            continue
        kept.append({**entry, "video_sponsorship_transcript": result})
        files.write_json(out_path, kept, indent=2, ensure_ascii=False)
    return out_path
