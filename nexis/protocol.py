"""Protocol-level constants and policy decisions for Nexisgen."""

from __future__ import annotations

from dataclasses import dataclass


PROTOCOL_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"

# Interval semantics (frozen for v1)
INTERVAL_MODE = "blocks"
# Chain interval start for miner R2 prefix / out/{interval_id}/ / validator windows: block % INTERVAL_LENGTH_BLOCKS == 0
INTERVAL_LENGTH_BLOCKS = 100
WEIGHT_SUBMISSION_INTERVAL_BLOCKS = 250
UPLOAD_DEADLINE_RESERVED_BLOCKS = 2

# Miner sampling semantics
MINER_SAMPLE_RATE = 1
MINER_SAMPLE_MIN = 2
MINER_SAMPLE_MAX = 35

# Per-miner row sampling semantics
ROW_SAMPLE_ALL_THRESHOLD = 3
ROW_SAMPLE_RATE = 0.20
ROW_SAMPLE_MAX = 3

# Scoring
SCORING_EXPONENT = 3
FAILURE_LOOKBACK_INTERVALS = 1

# Data policy
CLIP_DURATION_SEC = 5.0
MIN_CLIP_GAP_SEC = 5.0
# Lexical caption minimum: captions must be strictly longer than 20 words (platform short_caption gate).
MIN_CAPTION_WORDS = 25

# Miner captioner: if the model returns fewer than this many words, discard and skip the segment (no retry).
CAPTION_SKIP_IF_FEWER_WORDS = 20

# Miner prepare (mine_one_segment): advance to next source URL after this many consecutive merge/validation rejects.
PREP_MAX_CONSECUTIVE_REJECTS_PER_URL = 10


@dataclass(frozen=True)
class SoftFailurePolicy:
    """Threshold policy for soft checks."""

    threshold: float = 0.50


@dataclass(frozen=True)
class HardFailurePolicy:
    """Hard checks reject the interval immediately for that miner."""

    reject_on_first_violation: bool = True

