# Miner Guide

## Required `.env` Keys

- `BT_WALLET_NAME`
- `BT_WALLET_HOTKEY`
- `BT_WALLET_PATH`
- `HIPPIUS_BUCKET`
- `HIPPIUS_READ_ACCESS_KEY`
- `HIPPIUS_READ_SECRET_KEY`
- `HIPPIUS_WRITE_ACCESS_KEY`
- `HIPPIUS_WRITE_SECRET_KEY`
- `NEXIS_SOURCES_FILE`
- `NEXIS_BLOCK_POLL_SEC` (optional, default: `6`)
- `TARGET_RESOLUTION` (optional, default: `720`)
- `NEXIS_DATASET_SPEC_DEFAULT` (optional, default: `video_v1`)
- `NEXIS_MINER_ENABLED_SPECS` (optional, CSV, default: `video_v1`)

## Required System Tools

- `yt-dlp`
- `ffmpeg`
- `ffprobe`

## Commands

```bash
nexis commit-credentials
nexis mine
nexis mine --spec video_v1
# optional debug logging:
# nexis mine --debug
# optional poll override:
# nexis mine --poll-sec 4
```

`nexis mine` runs continuously. Every new 50-block interval, the miner
builds and uploads one interval package keyed by interval start block.
Captions are generated from timeline-sampled clip frames (default: 12 frames).
Stop with `Ctrl+C`.

