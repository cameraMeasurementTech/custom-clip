# Nexis CLI runbook

How to run each `nexis` command. From the repo root (with the package installed, e.g. `pip install -e ".[dev]"`):

```bash
nexis --help
nexis <command> --help
```

Configuration is read from `.env` in the current working directory (and defaults in `nexis/config.py`). Wallet and R2 variables must be set before miner/validator commands.

### Without the `nexis` console script

If `nexis` is missing or is an older install, use the same logic via `python -m` (install the package from this repo first, e.g. `pip install -e ".[dev]"`):

```bash
python3 -m nexis.tools.mine_prepare_main --workdir /data/pool --sources urls.txt \
  --openai-api-key "$OPENAI_API_KEY" --openai-api-keys-extra "sk-second,sk-third"

python3 -m nexis.tools.mine_upload_main --workdir /data/pool
# Repeats every 30m by default; chain interval id is computed from the current block.
# One upload then exit: --every once
# Override id: --interval-id 1234560
```

`mine_prepare_main` sets the given key flags in the process environment before loading settings (so they override `.env` for that run). `mine_upload_main` accepts `--every` (e.g. `30m`, `1h`, `once`) and optional `--interval-id`; use `--help` on each module.

---

## `nexis commit-credentials`

**Purpose:** Publish your miner’s R2 **read** credentials on-chain so validators can list and download your bucket.

**When:** Once per setup (or when keys change).

```bash
nexis commit-credentials
```

Uses `BT_WALLET_*`, `NEXIS_NETUID`, `BT_NETWORK`, and `R2_*` / bucket naming from `nexis/storage/r2.py`.

---

## `nexis mine`

**Purpose:** Long-running miner loop: follows the chain, builds 5 s clips, captions, and uploads to R2.

**Behavior (env-driven):**

- **`NEXIS_MINER_CONTINUOUS_PACK_MODE=true` (default):** Each poll builds **one** clip into `NEXIS_WORKDIR/miner_pending_records.jsonl` from `NEXIS_SOURCES_FILE`. On upload cadence (`NEXIS_MINER_UPLOAD_CADENCE_BLOCKS`, default = `INTERVAL_LENGTH_BLOCKS` in `nexis/protocol.py`), uploads **new** pending rows (deduped vs `miner_uploaded_clip_ids.txt`) under `{interval_id}/` on R2. **`interval_id`** is the chain interval start: `current_block - (current_block % INTERVAL_LENGTH_BLOCKS)` (a multiple of **100** with the current protocol).
- **`NEXIS_MINER_CONTINUOUS_PACK_MODE=false`:** Legacy: full batch per chain interval via `run_interval`.

**Examples:**

```bash
nexis mine
nexis mine --spec video_v1
nexis mine --debug
nexis mine --poll-sec 4
```

**Stop:** `Ctrl+C`.

**Related env:** `NEXIS_SOURCES_FILE`, `NEXIS_WORKDIR`, `NEXIS_MINER_CONTINUOUS_PACK_MODE`, `NEXIS_MINER_UPLOAD_CADENCE_BLOCKS`, `OPENAI_API_KEY`, `NEXIS_OPENAI_API_KEYS`, `NEXIS_CAPTION_OPENAI_RPM_PER_KEY`, `GEMINI_API_KEY`, etc.

---

## `nexis mine-prepare`

**Purpose:** **Offline / parallel prepare** — pop YouTube URLs from a **shared queue file**, download, clip, caption into a **shared workdir**. **Does not** upload to R2. Use `mine-upload` separately.

**Queue file:** The first non-empty, non-`#` line is **removed** when a worker claims a URL (POSIX lock: `<sources>.queue.lock`). Keep a backup of the list if needed.

**Workdir layout:** `clips/`, `frames/`, `miner_pending_records.jsonl`, per-worker `raw/wN/`, `cursors/miner_segment_cursor.wN.json`.

**OpenAI — one worker:**

```bash
nexis mine-prepare --workdir /data/pool --sources urls.txt --workers 1
```

Uses `OPENAI_API_KEY` + `NEXIS_OPENAI_API_KEYS` from `.env` (merged). Optional override:

```bash
nexis mine-prepare --workdir /data/pool --sources urls.txt --workers 1 \
  --thread-keys-file keys.txt
```

(first line of `keys.txt` = comma-separated keys for that worker)

**OpenAI — multiple workers (each line = one worker’s keys):**

`keys.txt` must have **exactly** `--workers` non-empty lines:

```text
sk-proj-aaa,sk-proj-bbb
sk-proj-ccc,sk-proj-ddd
sk-proj-eee,sk-proj-fff
```

```bash
nexis mine-prepare --workdir /data/pool --sources urls.txt --workers 3 \
  --thread-keys-file keys.txt
```

**Gemini (single worker only):**

```bash
nexis mine-prepare --workdir /data/pool --sources urls.txt --workers 1
# requires GEMINI_API_KEY; do not pass --thread-keys-file
```

**Other options:**

| Option | Meaning |
|--------|--------|
| `--max-segments-per-worker N` | Stop each worker after `N` clips (`0` = until queue empty). |
| `--rpm-per-key N` | Override `NEXIS_CAPTION_OPENAI_RPM_PER_KEY` (OpenAI spacing). |
| `--spec video_v1` | Dataset spec (must be enabled in `NEXIS_MINER_ENABLED_SPECS`). |
| `--debug` | Verbose logging. |

**Safety:** Prefer **not** running `mine-prepare` while `mine-upload` is deleting files from the same `--workdir` with default delete behavior.

---

## `nexis mine-upload`

**Purpose:** Upload **pending** clips from a workdir to R2 for a given chain **`interval_id`** (prefix `{interval_id}/`). Only rows whose `clip_id` is **not** already in `miner_uploaded_clip_ids.txt` are packed (same logic as the continuous miner).

**Prerequisites:** Same wallet/R2 `.env` as `nexis mine`; `commit-credentials` should already be done for validators.

```bash
nexis mine-upload --workdir /data/pool --interval-id 12345600
```

**Keep local copies after upload (debug):**

```bash
nexis mine-upload --workdir /data/pool --interval-id 12345600 --keep-local
```

Default (`--delete-local`): after a **successful** upload, removes uploaded `clips/` and `frames/` assets, caption scratch dirs under `frames/{clip_id}/`, and local `out/{interval_id}/`.

**Before upload:** If `{interval_id}/manifest.json` already exists on R2, the pack step **waits** (default **1200 s / 20 m**) and retries; in `nexis mine` continuous mode and in `mine_upload_main` without `--interval-id`, the interval id is **re-read from the chain** after each wait. Set `NEXIS_MINER_UPLOAD_MANIFEST_BUSY_WAIT_SEC` (use `0` to disable the check).

**R2 retention:** After each successful pack upload, the code can delete an **older** interval folder on the bucket (default: the prefix from **two uploads ago**), using `miner_r2_upload_interval_history.json` in the workdir. Configure with `NEXIS_MINER_R2_PRUNE_OLD_PREFIX` and `NEXIS_MINER_R2_PRUNE_UPLOADS_AGO`. The `python -m nexis.tools.mine_upload_main` entrypoint also supports `--no-r2-prune` and `--r2-prune-uploads-ago`.

**Other options:** `--spec video_v1`, `--debug`.

---

## `nexis validate`

**Purpose:** Validator loop: discover miners, download interval packages, run checks, score, submit weights.

```bash
nexis validate
nexis validate --debug
nexis validate --poll-sec 6
nexis validate --specs video_v1
nexis validate --blacklist-file /path/to/list.txt
nexis validate --exclude-hotkeys hk1,hk2
```

Requires validator wallet, R2 read access to miners, and LLM keys if semantic/category checks are enabled. See `docs/validator.md`.

---

## `nexis validate-source-auth`

**Purpose:** Validator loop focused on **source authenticity** and reporting to the evidence API.

**Requires:** `NEXIS_VALIDATION_API_URL` set in `.env`.

```bash
nexis validate-source-auth
nexis validate-source-auth --debug --poll-sec 10
```

Same style of options as `validate` where applicable (`--blacklist-file`, `--exclude-hotkeys`, `--specs`).

---

## `nexis sync-owner-datasets`

**Purpose:** Owner-validator helper: periodically copies accepted miner assets into the owner dataset bucket (see README “Owner-validator”).

```bash
nexis sync-owner-datasets
nexis sync-owner-datasets --poll-sec 60
nexis sync-owner-datasets --blacklist-file /path/to/list.txt
```

Runs until `Ctrl+C`.

---

## Quick reference

| Command | Typical use |
|---------|-------------|
| `commit-credentials` | Miner: one-time (or key rotation) on-chain read creds |
| `mine` | Miner: production loop with chain + R2 |
| `mine-prepare` | Miner: parallel CPU/IO prep into a shared dir, no R2 |
| `mine-upload` | Miner: push prepared pending to R2 for one `interval_id` |
| `validate` | Validator: full validation loop |
| `validate-source-auth` | Validator: source-auth + API reporting |
| `sync-owner-datasets` | Owner-validator: dataset copy worker |

For dataset schema and validation stages, see `docs/miner.md`, `docs/validator.md`, and `docs/dataset-schema.md`.
