import os
import random
import time
from pathlib import Path
import requests

# Set YOUTUBE_API_KEY in the environment (do not commit keys). Revoke any key that was ever committed.
API_KEY = os.environ.get("YOUTUBE_API_KEY", "AIzaSyCZge7BfkfnYhE7Jc8y1hP2xpFcCPrjIsQ")

# When YOUTUBE_SEARCH_QUERY is not set, each run picks one variant at random (different results per run).
# Biased toward NatureCategoryChecker + CaptionSemanticChecker (scenic B-roll, not vlog/urban).
DEFAULT_QUERY_VARIANTS = [
    "natural scenery landscape mountains forest lake wilderness aerial timelapse b-roll"
]

QUERY = "natural scenery landscape b-roll"

MAX_RESULTS = 50  # per request (max 50)
TOTAL_VIDEOS = 100  # how many you want

BASE_URL = "https://www.googleapis.com/youtube/v3/search"

# API search order — random each run mixes ranking so you are not stuck on the same first page.
_SEARCH_ORDERS = ("relevance", "date", "viewCount", "rating", "title", "videoCount")

# Search filters (require type=video):
# - videoDefinition=high → HD only: YouTube documents this as at least 720p (1280×720-class). It can still be 1080p/4K etc.;
#   the Data API has no "exactly 720p" or "max 720p" filter — only high vs standard vs any.
# - videoDuration=long → longer than 20 minutes (API bucket for "long" content).
# VIDEO_DEFINITION = "high"  # "any" | "high" | "standard"
VIDEO_DURATION = "medium"  # "any" | "short" (<4m) | "medium" (4–20m) | "long" (>20m)


def resolve_search_query() -> str:
    """Fixed query from env, or a random variant so each run hits a different slice of results."""
    env_q = os.environ.get("YOUTUBE_SEARCH_QUERY", "").strip()
    if env_q:
        return env_q
    return random.choice(DEFAULT_QUERY_VARIANTS)


def pick_search_order() -> str:
    return random.choice(_SEARCH_ORDERS)


def default_seen_ids_path(sources_path: Path | None = None) -> Path:
    base = sources_path.parent if sources_path is not None else Path(__file__).resolve().parent / "submit"
    return base / "youtube_seen_video_ids.txt"


def load_seen_video_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_seen_video_ids(path: Path, video_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_seen_video_ids(path)
    new = [v for v in video_ids if v not in existing]
    if not new:
        return
    with path.open("a", encoding="utf-8") as f:
        for v in new:
            f.write(f"{v}\n")


def get_youtube_urls(
    api_key: str,
    query: str,
    total_videos: int,
    *,
    exclude_video_ids: set[str] | None = None,
    order: str | None = None,
) -> tuple[list[str], list[str], int, str]:
    """Return (watch_urls, video_ids, skipped_seen_count, order_used)."""
    if not api_key:
        raise ValueError(
            "Missing YOUTUBE_API_KEY. Example: set YOUTUBE_API_KEY in your environment."
        )
    if not (query or "").strip():
        raise ValueError("Search query is empty; set YOUTUBE_SEARCH_QUERY or fix DEFAULT_QUERY_VARIANTS.")

    exclude = set(exclude_video_ids or ())
    order = order or pick_search_order()
    if order not in _SEARCH_ORDERS:
        order = "relevance"

    video_urls: list[str] = []
    collected_ids: list[str] = []
    next_page_token = None
    skipped = 0

    while len(video_urls) < total_videos:
        params = {
            "part": "snippet",
            "q": QUERY,
            "type": "video",
            "maxResults": MAX_RESULTS,
            "key": api_key,
            # "videoDefinition": "high",
            "videoDuration": VIDEO_DURATION,
            "order": order,
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise RuntimeError(data["error"])

        for item in data.get("items", []):
            video_id = item["id"]["videoId"]
            if video_id in exclude:
                skipped += 1
                continue
            exclude.add(video_id)
            url = f"https://www.youtube.com/watch?v={video_id}"
            video_urls.append(url)
            collected_ids.append(video_id)

            if len(video_urls) >= total_videos:
                break

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(1)  # avoid quota issues

    print(collected_ids)
    return video_urls, collected_ids, skipped, order


def save_to_sources(urls, path: str | Path | None = None) -> Path:
    """Append one YouTube URL per line; existing file content is left unchanged."""
    out = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parent / "submit" / "search_results.txt"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if not urls:
        return out

    prefix = ""
    if out.exists() and out.stat().st_size > 0:
        with out.open("rb") as rf:
            rf.seek(-1, os.SEEK_END)
            if rf.read(1) != b"\n":
                prefix = "\n"

    with out.open("a", encoding="utf-8", newline="\n") as f:
        f.write(prefix)
        for url in urls:
            f.write(url)
            f.write("\n")
    return out


if __name__ == "__main__":
    sources_out = Path(__file__).resolve().parent / "submit" / "search.txt"
    seen_path = Path(
        os.environ.get(
            "YOUTUBE_SEEN_IDS_FILE",
            str(default_seen_ids_path(sources_out)),
        )
    )
    query = resolve_search_query()
    order = pick_search_order()
    known = load_seen_video_ids(seen_path)

    urls, ids_new, skipped, order_used = get_youtube_urls(
        API_KEY,
        query,
        TOTAL_VIDEOS,
        exclude_video_ids=known,
        order=order,
    )
    out_path = save_to_sources(urls, sources_out)
    append_seen_video_ids(seen_path, ids_new)

    print(
        f"query={query!r} order={order_used} skipped_already_seen={skipped} "
        f"new={len(urls)} -> {out_path} (seen log: {seen_path})"
    )
