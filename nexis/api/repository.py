"""Persistence layer for validation evidence API."""

from __future__ import annotations

from typing import Any

from .db import Database
from .schemas import DecisionIngestItem


class ValidationEvidenceRepository:
    """Read/write operations for validator interval evidence."""

    def __init__(self, db: Database):
        self._db = db

    async def ensure_schema(self) -> None:
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS validation_results (
                validator_hotkey TEXT NOT NULL,
                interval_id BIGINT NOT NULL,
                miner_hotkey TEXT NOT NULL,
                accepted BOOLEAN NOT NULL,
                failure_reasons TEXT[] NOT NULL DEFAULT '{}',
                record_count INTEGER NOT NULL DEFAULT 0,
                global_overlap_pruned_count INTEGER NOT NULL DEFAULT 0,
                cross_miner_overlap_pruned_count INTEGER NOT NULL DEFAULT 0,
                signature TEXT NOT NULL,
                signature_timestamp BIGINT NOT NULL,
                signature_nonce TEXT NOT NULL,
                body_sha256 TEXT NOT NULL,
                received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (validator_hotkey, interval_id, miner_hotkey)
            )
            """
        )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_validation_results_validator_interval
                ON validation_results (validator_hotkey, interval_id)
            """
        )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_validation_results_interval
                ON validation_results (interval_id)
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS validator_request_nonces (
                validator_hotkey TEXT NOT NULL,
                nonce TEXT NOT NULL,
                signature_timestamp BIGINT NOT NULL,
                received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (validator_hotkey, nonce)
            )
            """
        )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_validator_request_nonces_received_at
                ON validator_request_nonces (received_at)
            """
        )

    async def register_nonce_once(
        self,
        *,
        validator_hotkey: str,
        nonce: str,
        signature_timestamp: int,
        max_age_sec: int,
    ) -> bool:
        # Best-effort cleanup to keep nonce table bounded.
        await self._db.execute(
            """
            DELETE FROM validator_request_nonces
            WHERE received_at < NOW() - ($1::BIGINT * INTERVAL '1 second')
            """,
            int(max(max_age_sec, 1)),
        )
        inserted = await self._db.fetchval(
            """
            INSERT INTO validator_request_nonces (
                validator_hotkey,
                nonce,
                signature_timestamp
            )
            VALUES ($1, $2, $3)
            ON CONFLICT (validator_hotkey, nonce) DO NOTHING
            RETURNING 1
            """,
            validator_hotkey,
            nonce,
            int(signature_timestamp),
        )
        return inserted == 1

    async def upsert_interval_decisions(
        self,
        *,
        validator_hotkey: str,
        interval_id: int,
        decisions: list[DecisionIngestItem],
        signature: str,
        signature_timestamp: int,
        signature_nonce: str,
        body_sha256: str,
    ) -> int:
        values = [
            (
                validator_hotkey,
                int(interval_id),
                decision.miner_hotkey,
                decision.accepted,
                decision.failures,
                int(decision.record_count),
                int(decision.global_overlap_pruned_count),
                int(decision.cross_miner_overlap_pruned_count),
                signature,
                int(signature_timestamp),
                signature_nonce,
                body_sha256,
            )
            for decision in decisions
        ]
        inserted = 0
        for args in values:
            result = await self._db.fetchval(
                """
                INSERT INTO validation_results (
                    validator_hotkey,
                    interval_id,
                    miner_hotkey,
                    accepted,
                    failure_reasons,
                    record_count,
                    global_overlap_pruned_count,
                    cross_miner_overlap_pruned_count,
                    signature,
                    signature_timestamp,
                    signature_nonce,
                    body_sha256
                )
                VALUES (
                    $1, $2, $3, $4, $5,
                    $6, $7, $8, $9, $10,
                    $11, $12
                )
                ON CONFLICT (validator_hotkey, interval_id, miner_hotkey)
                DO NOTHING
                RETURNING 1
                """,
                *args,
            )
            if result == 1:
                inserted += 1
        return inserted

    async def get_interval_decisions(
        self,
        *,
        validator_hotkey: str,
        interval_id: int,
    ) -> list[dict[str, Any]]:
        rows = await self._db.fetch(
            """
            SELECT
                interval_id,
                validator_hotkey,
                miner_hotkey,
                accepted,
                failure_reasons,
                record_count,
                global_overlap_pruned_count,
                cross_miner_overlap_pruned_count,
                signature,
                signature_timestamp,
                signature_nonce,
                body_sha256,
                received_at
            FROM validation_results
            WHERE validator_hotkey = $1
              AND interval_id = $2
            ORDER BY miner_hotkey ASC
            """,
            validator_hotkey,
            int(interval_id),
        )
        payload: list[dict[str, Any]] = []
        for row in rows:
            payload.append(
                {
                    "interval_id": int(row["interval_id"]),
                    "validator_hotkey": str(row["validator_hotkey"]),
                    "miner_hotkey": str(row["miner_hotkey"]),
                    "accepted": bool(row["accepted"]),
                    "failures": [str(item) for item in row["failure_reasons"] or []],
                    "record_count": int(row["record_count"]),
                    "global_overlap_pruned_count": int(row["global_overlap_pruned_count"]),
                    "cross_miner_overlap_pruned_count": int(
                        row["cross_miner_overlap_pruned_count"]
                    ),
                    "signature": str(row["signature"]),
                    "timestamp": int(row["signature_timestamp"]),
                    "nonce": str(row["signature_nonce"]),
                    "body_sha256": str(row["body_sha256"]),
                    "received_at": row["received_at"],
                }
            )
        return payload

    async def get_decisions_in_interval_range(
        self,
        *,
        start_interval_id: int,
        end_interval_id: int,
    ) -> list[dict[str, Any]]:
        rows = await self._db.fetch(
            """
            SELECT
                interval_id,
                validator_hotkey,
                miner_hotkey,
                accepted,
                failure_reasons,
                record_count,
                global_overlap_pruned_count,
                cross_miner_overlap_pruned_count,
                received_at
            FROM validation_results
            WHERE interval_id >= $1
              AND interval_id <= $2
            ORDER BY interval_id DESC, received_at DESC, validator_hotkey ASC, miner_hotkey ASC
            """,
            int(start_interval_id),
            int(end_interval_id),
        )
        payload: list[dict[str, Any]] = []
        for row in rows:
            payload.append(
                {
                    "interval_id": int(row["interval_id"]),
                    "validator_hotkey": str(row["validator_hotkey"]),
                    "miner_hotkey": str(row["miner_hotkey"]),
                    "accepted": bool(row["accepted"]),
                    "failures": [str(item) for item in row["failure_reasons"] or []],
                    "record_count": int(row["record_count"]),
                    "global_overlap_pruned_count": int(row["global_overlap_pruned_count"]),
                    "cross_miner_overlap_pruned_count": int(
                        row["cross_miner_overlap_pruned_count"]
                    ),
                    "received_at": row["received_at"],
                }
            )
        return payload

