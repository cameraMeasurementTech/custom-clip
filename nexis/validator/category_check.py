"""Category validation checks for sampled rows."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import ClipRecord

_DEFAULT_STRICT_MODEL = "gemini-3.1-flash-lite-preview"
_CATEGORY_NATURE = "nature"
_STRICT_ALLOWED_WINNERS = {
    "nature",
    "people",
    "animal",
    "vehicle",
    "urban",
    "indoor",
    "other",
}
_STRICT_PROMPT = """
You are validating whether a clip belongs to:
Nature / landscape / scenery.

You are given the middle 3 frames from a 5-second clip.

For each frame, decide which category is dominant:
- nature
- people
- animal
- vehicle
- urban
- indoor
- other

Strict rule:
- PASS for nature only when natural scenery is the main subject.
- If a person is central and dominant, winner must not be nature.
- If an animal is the main subject, winner must not be nature.
- If a vehicle, city/urban scene, or indoor scene dominates, winner must not be nature.

Return JSON only:
{
  "frames": [
    {
      "frame_index": 0,
      "winner": "nature|people|animal|vehicle|urban|indoor|other",
      "nature_score": 0.0,
      "people_score": 0.0,
      "animal_score": 0.0,
      "vehicle_score": 0.0,
      "urban_score": 0.0,
      "indoor_score": 0.0
    },
    {
      "frame_index": 1,
      "winner": "nature|people|animal|vehicle|urban|indoor|other",
      "nature_score": 0.0,
      "people_score": 0.0,
      "animal_score": 0.0,
      "vehicle_score": 0.0,
      "urban_score": 0.0,
      "indoor_score": 0.0
    },
    {
      "frame_index": 2,
      "winner": "nature|people|animal|vehicle|urban|indoor|other",
      "nature_score": 0.0,
      "people_score": 0.0,
      "animal_score": 0.0,
      "vehicle_score": 0.0,
      "urban_score": 0.0,
      "indoor_score": 0.0
    }
  ]
}
"""

NATURE_WORDS = {
    "nature",
    "landscape",
    "scenery",
    "scenic",
    "forest",
    "mountain",
    "lake",
    "river",
    "waterfall",
    "beach",
    "coast",
    "coastline",
    "ocean",
    "sea",
    "desert",
    "valley",
    "cliff",
    "canyon",
    "sky",
    "clouds",
    "sunset",
    "sunrise",
    "snow",
    "hill",
    "hills",
    "trees",
    "woods",
    "meadow",
    "field",
    "shore",
    "rocky shore",
    "natural",
    "outdoor",
    "outdoors",
}
PEOPLE_WORDS = {
    "person",
    "people",
    "man",
    "woman",
    "boy",
    "girl",
    "hiker",
    "tourist",
    "selfie",
    "portrait",
    "speaker",
    "talking to camera",
    "vlogger",
    "vlog",
    "human",
    "couple",
    "friends",
    "family",
}
ANIMAL_WORDS = {
    "animal",
    "dog",
    "cat",
    "bird",
    "deer",
    "horse",
    "cow",
    "elephant",
    "lion",
    "tiger",
    "bear",
    "pet",
    "wildlife",
    "monkey",
    "fox",
}
VEHICLE_WORDS = {
    "car",
    "truck",
    "bus",
    "bike",
    "bicycle",
    "motorcycle",
    "boat",
    "ship",
    "train",
    "airplane",
    "plane",
    "helicopter",
    "vehicle",
    "driving",
}
URBAN_WORDS = {
    "city",
    "street",
    "urban",
    "building",
    "buildings",
    "skyscraper",
    "traffic",
    "crosswalk",
    "storefront",
    "downtown",
    "road",
    "highway",
    "bridge",
    "architecture",
    "parking lot",
}
INDOOR_WORDS = {
    "indoor",
    "inside",
    "room",
    "bedroom",
    "kitchen",
    "office",
    "hallway",
    "studio",
    "living room",
    "restaurant",
    "cafe",
    "shop",
}

_STRONG_NATURE_PHRASES = {
    "natural landscape",
    "scenic landscape",
    "mountain landscape",
    "nature scenery",
    "wide landscape shot",
    "natural scenery",
    "the clip shows a forest",
    "the clip shows a mountain",
    "the scene shows a lake",
    "the main subject is nature",
    "the main subject is a landscape",
    "the main focus is natural scenery",
}


@dataclass
class CaptionCheckResult:
    main_subject: str
    nature_score: float
    people_score: float
    animal_score: float
    vehicle_score: float
    urban_score: float
    indoor_score: float
    conflict: bool
    reason: str


@dataclass
class FrameResult:
    frame_index: int
    winner: str
    nature_score: float
    people_score: float
    animal_score: float
    vehicle_score: float
    urban_score: float
    indoor_score: float


@dataclass
class StrictPassResult:
    frames: list[FrameResult]


@dataclass
class FinalDecision:
    decision: str
    stage: str
    reason: str
    nature_score: float
    rival_score: float
    margin: float


def clamp_score(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def count_keyword_hits(text: str, vocab: set[str]) -> int:
    hits = 0
    for word in vocab:
        if _contains_term(text, word):
            hits += 1
    return hits


def _contains_term(text: str, term: str) -> bool:
    if " " in term:
        return term in text
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def infer_main_subject_from_caption(caption: str, scores: dict[str, float]) -> str:
    _ = caption
    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    if best_score < 0.20:
        return "other"
    return best_label


def caption_gate_check_nature(miner_caption: str) -> CaptionCheckResult:
    text = normalize_text(miner_caption)

    nature_hits = count_keyword_hits(text, NATURE_WORDS)
    people_hits = count_keyword_hits(text, PEOPLE_WORDS)
    animal_hits = count_keyword_hits(text, ANIMAL_WORDS)
    vehicle_hits = count_keyword_hits(text, VEHICLE_WORDS)
    urban_hits = count_keyword_hits(text, URBAN_WORDS)
    indoor_hits = count_keyword_hits(text, INDOOR_WORDS)

    nature_score = clamp_score(0.12 * nature_hits)
    people_score = clamp_score(0.18 * people_hits)
    animal_score = clamp_score(0.18 * animal_hits)
    vehicle_score = clamp_score(0.18 * vehicle_hits)
    urban_score = clamp_score(0.18 * urban_hits)
    indoor_score = clamp_score(0.18 * indoor_hits)

    if any(phrase in text for phrase in _STRONG_NATURE_PHRASES):
        nature_score = clamp_score(nature_score + 0.20)

    if (
        "person in front of" in text
        or "woman in front of" in text
        or "man in front of" in text
    ):
        people_score = clamp_score(people_score + 0.25)
        nature_score = clamp_score(nature_score - 0.20)

    if "driving through" in text or "car on" in text or "vehicle on" in text:
        vehicle_score = clamp_score(vehicle_score + 0.25)
        nature_score = clamp_score(nature_score - 0.20)

    if (
        "close-up of a deer" in text
        or "close-up of a bird" in text
        or "close-up of an animal" in text
    ):
        animal_score = clamp_score(animal_score + 0.25)
        nature_score = clamp_score(nature_score - 0.20)

    rival_max = max(people_score, animal_score, vehicle_score, urban_score, indoor_score)
    conflict = rival_max >= 0.45

    scores = {
        "nature": nature_score,
        "people": people_score,
        "animal": animal_score,
        "vehicle": vehicle_score,
        "urban": urban_score,
        "indoor": indoor_score,
    }
    main_subject = infer_main_subject_from_caption(text, scores)
    reason = (
        f"caption-based scores: nature={nature_score:.2f}, people={people_score:.2f}, "
        f"animal={animal_score:.2f}, vehicle={vehicle_score:.2f}, "
        f"urban={urban_score:.2f}, indoor={indoor_score:.2f}"
    )
    return CaptionCheckResult(
        main_subject=main_subject,
        nature_score=nature_score,
        people_score=people_score,
        animal_score=animal_score,
        vehicle_score=vehicle_score,
        urban_score=urban_score,
        indoor_score=indoor_score,
        conflict=conflict,
        reason=reason,
    )


def best_rival_score_from_caption(result: CaptionCheckResult) -> float:
    return max(
        result.people_score,
        result.animal_score,
        result.vehicle_score,
        result.urban_score,
        result.indoor_score,
    )


def caption_gate_decision_from_caption(result: CaptionCheckResult) -> str:
    rival = best_rival_score_from_caption(result)
    margin = result.nature_score - rival
    if (
        result.main_subject == _CATEGORY_NATURE
        and result.nature_score >= 0.92
        and not result.conflict
        and margin >= 0.35
    ):
        return "accept"
    if (
        result.nature_score < 0.55
        or (result.main_subject != _CATEGORY_NATURE and rival > result.nature_score + 0.10)
        or rival >= 0.75
    ):
        return "reject"
    return "borderline"


def get_middle_three_frame_paths(frame_paths: list[Path]) -> list[Path] | None:
    existing = [path for path in frame_paths if path.exists()]
    if not existing:
        return None
    unique: list[Path] = []
    seen: set[str] = set()
    for path in existing:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    if len(unique) < 3:
        return None
    if len(unique) == 3:
        return unique
    start = max(0, min(len(unique) - 3, (len(unique) // 2) - 1))
    return unique[start : start + 3]


def parse_strict_pass(data: dict[str, Any]) -> StrictPassResult | None:
    frames_raw = data.get("frames", [])
    if not isinstance(frames_raw, list) or len(frames_raw) != 3:
        return None
    frames: list[FrameResult] = []
    for item in frames_raw:
        if not isinstance(item, dict):
            return None
        winner = str(item.get("winner", "")).strip().lower()
        if winner not in _STRICT_ALLOWED_WINNERS:
            return None
        frames.append(
            FrameResult(
                frame_index=int(item.get("frame_index", 0)),
                winner=winner,
                nature_score=clamp_score(item.get("nature_score", 0.0)),
                people_score=clamp_score(item.get("people_score", 0.0)),
                animal_score=clamp_score(item.get("animal_score", 0.0)),
                vehicle_score=clamp_score(item.get("vehicle_score", 0.0)),
                urban_score=clamp_score(item.get("urban_score", 0.0)),
                indoor_score=clamp_score(item.get("indoor_score", 0.0)),
            )
        )
    return StrictPassResult(frames=frames)


def strict_pass_decision(strict_result: StrictPassResult) -> FinalDecision:
    nature_scores = [f.nature_score for f in strict_result.frames]
    people_scores = [f.people_score for f in strict_result.frames]
    animal_scores = [f.animal_score for f in strict_result.frames]
    vehicle_scores = [f.vehicle_score for f in strict_result.frames]
    urban_scores = [f.urban_score for f in strict_result.frames]
    indoor_scores = [f.indoor_score for f in strict_result.frames]

    avg_nature = sum(nature_scores) / len(nature_scores)
    avg_people = sum(people_scores) / len(people_scores)
    avg_animal = sum(animal_scores) / len(animal_scores)
    avg_vehicle = sum(vehicle_scores) / len(vehicle_scores)
    avg_urban = sum(urban_scores) / len(urban_scores)
    avg_indoor = sum(indoor_scores) / len(indoor_scores)

    best_rival = max(avg_people, avg_animal, avg_vehicle, avg_urban, avg_indoor)
    margin = avg_nature - best_rival
    nature_wins = sum(1 for frame in strict_result.frames if frame.winner == _CATEGORY_NATURE)
    reason = (
        f"Nature wins {nature_wins}/3 middle frames, "
        f"avg_nature={avg_nature:.2f}, margin={margin:.2f}"
    )

    if nature_wins >= 2 and avg_nature >= 0.72 and margin >= 0.12:
        return FinalDecision(
            decision="accept",
            stage="strict",
            reason=reason,
            nature_score=avg_nature,
            rival_score=best_rival,
            margin=margin,
        )

    return FinalDecision(
        decision="reject",
        stage="strict",
        reason=reason,
        nature_score=avg_nature,
        rival_score=best_rival,
        margin=margin,
    )


class NatureCategoryChecker:
    """Category checker with caption gate and strict vision pass."""

    def __init__(
        self,
        *,
        enabled: bool,
        api_key: str,
        timeout_sec: int,
        max_samples: int,
        base_url: str | None,
        model: str = _DEFAULT_STRICT_MODEL,
    ):
        self._enabled = enabled
        self._api_key = api_key.strip()
        self._timeout_sec = timeout_sec
        self._max_samples = max(0, max_samples)
        self._base_url = base_url
        self._model = model

    @property
    def active(self) -> bool:
        return self._enabled and self._max_samples > 0

    def check(
        self,
        *,
        sampled: list[ClipRecord],
        frame_paths_by_clip_id: dict[str, list[Path]],
    ) -> list[str]:
        if not self.active:
            return []

        failures: list[str] = []
        checked = 0
        client = None
        for row in sampled:
            if checked >= self._max_samples:
                break
            caption_gate_result = caption_gate_check_nature(row.caption)
            caption_gate_decision = caption_gate_decision_from_caption(caption_gate_result)
            if caption_gate_decision == "reject":
                failures.append(f"category_caption_reject:{row.clip_id}")
                checked += 1
                continue
            if caption_gate_decision == "accept":
                checked += 1
                continue

            middle_three = get_middle_three_frame_paths(frame_paths_by_clip_id.get(row.clip_id, []))
            if middle_three is None:
                failures.append(f"category_strict_frames_missing:{row.clip_id}")
                checked += 1
                continue

            if not self._api_key:
                failures.append(f"category_strict_api_key_missing:{row.clip_id}")
                checked += 1
                continue
            if client is None:
                client = self._build_client()
                if client is None:
                    failures.append(f"category_strict_client_unavailable:{row.clip_id}")
                    checked += 1
                    continue
            parsed = self._run_strict_pass(client=client, frame_paths=middle_three)
            if parsed is None:
                failures.append(f"category_strict_response_invalid:{row.clip_id}")
                checked += 1
                continue
            decision = strict_pass_decision(parsed)
            if decision.decision == "reject":
                failures.append(f"category_strict_reject:{row.clip_id}")
            checked += 1
        return failures

    def _build_client(self) -> object | None:
        try:
            from openai import OpenAI

            kwargs: dict[str, Any] = {
                "api_key": self._api_key,
                "timeout": self._timeout_sec,
            }
            if self._base_url:
                kwargs["base_url"] = self._base_url
            return OpenAI(**kwargs)
        except Exception:
            return None

    def _run_strict_pass(self, *, client: object, frame_paths: list[Path]) -> StrictPassResult | None:
        content: list[dict[str, Any]] = [{"type": "text", "text": _STRICT_PROMPT}]
        for frame_path in frame_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._frame_data_uri(frame_path)},
                }
            )
        try:
            response = client.chat.completions.create(  # type: ignore[attr-defined]
                model=self._model,
                messages=[{"role": "user", "content": content}],
                max_tokens=320,
            )
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            text = getattr(message, "content", "") if message is not None else ""
            return self._parse_strict_text(str(text))
        except Exception:
            return None

    def _parse_strict_text(self, output_text: str) -> StrictPassResult | None:
        text = output_text.strip()
        if not text:
            return None
        try:
            data = json.loads(text)
            return parse_strict_pass(data)
        except Exception:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            return parse_strict_pass(data)
        except Exception:
            return None

    def _frame_data_uri(self, frame_path: Path) -> str:
        payload = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"
