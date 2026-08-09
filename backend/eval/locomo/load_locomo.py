#!/usr/bin/env python3
"""Download/cache official LOCOMO JSON and expose normalized samples."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

LOCOMO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(LOCOMO_DIR, "data")
DEFAULT_CACHE_PATH = os.path.join(DATA_DIR, "locomo10.json")
LOCOMO_RAW_URL = (
    "https://raw.githubusercontent.com/snap-research/LoCoMo/main/data/locomo10.json"
)


def _session_keys(conversation: Dict[str, Any]) -> List[str]:
    keys = []
    for key in conversation:
        if not key.startswith("session_"):
            continue
        suffix = key[len("session_") :]
        if suffix.isdigit():
            keys.append(key)
    return sorted(keys, key=lambda name: int(name.split("_")[1]))


def download_locomo(cache_path: str = DEFAULT_CACHE_PATH, force: bool = False) -> str:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path) and not force:
        return cache_path
    with urllib.request.urlopen(LOCOMO_RAW_URL, timeout=120) as response:
        payload = response.read()
    with open(cache_path, "wb") as handle:
        handle.write(payload)
    return cache_path


def load_locomo_samples(cache_path: str = DEFAULT_CACHE_PATH, download: bool = True) -> List[Dict[str, Any]]:
    path = cache_path
    if download and not os.path.exists(path):
        path = download_locomo(path)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Expected LOCOMO JSON to be a list of samples")
    return data


def sample_stats(sample: Dict[str, Any]) -> Dict[str, Any]:
    conversation = sample.get("conversation") or {}
    session_keys = _session_keys(conversation)
    turn_count = sum(len(conversation.get(key) or []) for key in session_keys)
    qa_pairs = sample.get("qa") or []
    return {
        "sample_id": sample.get("sample_id"),
        "session_count": len(session_keys),
        "turn_count": turn_count,
        "qa_count": len(qa_pairs),
        "speaker_a": conversation.get("speaker_a"),
        "speaker_b": conversation.get("speaker_b"),
    }


def select_stratified_subset(
    samples: List[Dict[str, Any]],
    target_n: int = 20,
) -> List[Dict[str, Any]]:
    """Pick up to target_n conversations stratified by dialogue length (short/medium/long)."""
    if not samples:
        return []
    if len(samples) <= target_n:
        return list(samples)

    ranked = sorted(samples, key=lambda item: sample_stats(item)["turn_count"])
    bucket_size = max(1, len(ranked) // 3)
    short = ranked[:bucket_size]
    medium = ranked[bucket_size : bucket_size * 2]
    long = ranked[bucket_size * 2 :]
    buckets = [short, medium, long]
    per_bucket = max(1, target_n // 3)
    remainder = target_n - per_bucket * 3

    chosen: List[Dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        quota = per_bucket + (1 if index < remainder else 0)
        chosen.extend(bucket[:quota])
    return chosen[:target_n]


def parse_sessions(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    conversation = sample.get("conversation") or {}
    speaker_a = conversation.get("speaker_a") or "speaker_a"
    speaker_b = conversation.get("speaker_b") or "speaker_b"
    sessions: List[Dict[str, Any]] = []
    for key in _session_keys(conversation):
        session_num = int(key.split("_")[1])
        date_key = f"session_{session_num}_date_time"
        turns_raw = conversation.get(key) or []
        turns: List[Dict[str, str]] = []
        for turn in turns_raw:
            speaker = str(turn.get("speaker") or "").strip()
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            role = "user" if speaker == speaker_a else "assistant"
            turns.append(
                {
                    "role": role,
                    "content": text,
                    "speaker": speaker,
                    "dia_id": str(turn.get("dia_id") or ""),
                }
            )
        sessions.append(
            {
                "session_key": key,
                "session_num": session_num,
                "date_time": conversation.get(date_key),
                "turns": turns,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
            }
        )
    return sessions


def normalize_qa_pairs(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    for index, item in enumerate(sample.get("qa") or []):
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if not question or not answer:
            continue
        evidence = item.get("evidence") or []
        session_ids = evidence_session_ids(evidence)
        pairs.append(
            {
                "qa_index": index,
                "question": question,
                "answer": answer,
                "category": item.get("category"),
                "evidence": evidence,
                "evidence_session_ids": session_ids,
                "max_evidence_session": max(session_ids) if session_ids else None,
                "fact_type": classify_fact_type(question, answer, session_ids),
            }
        )
    return pairs


_EVIDENCE_RE = re.compile(r"^D(\d+)(?::\d+)?$", re.IGNORECASE)
_DATE_HINT_RE = re.compile(
    r"\b(when|date|year|month|january|february|march|april|may|june|july|"
    r"august|september|october|november|december|\d{4})\b",
    re.IGNORECASE,
)
_PREF_HINT_RE = re.compile(
    r"\b(like|prefer|favorite|favourite|enjoy|hate|dislike|hobby|destress|"
    r"passion|love to|interested)\b",
    re.IGNORECASE,
)


def parse_evidence_ref(ref: Any) -> Optional[int]:
    """Return dialogue/session number from LoCoMo evidence like 'D1:2' or 'D3'."""
    text = str(ref or "").strip()
    match = _EVIDENCE_RE.match(text)
    if not match:
        # Sometimes evidence is plain 'D1:2' inside longer strings
        found = re.search(r"D(\d+)", text, flags=re.IGNORECASE)
        return int(found.group(1)) if found else None
    return int(match.group(1))


def evidence_session_ids(evidence: Any) -> List[int]:
    ids: List[int] = []
    for item in evidence or []:
        session_id = parse_evidence_ref(item)
        if session_id is not None:
            ids.append(session_id)
    return sorted(set(ids))


def qa_evidence_covered(qa: Dict[str, Any], max_closed_sessions: int) -> bool:
    """True iff all evidence sessions are within 1..max_closed_sessions."""
    session_ids = qa.get("evidence_session_ids")
    if session_ids is None:
        session_ids = evidence_session_ids(qa.get("evidence") or [])
    if not session_ids:
        return False
    return max(session_ids) <= int(max_closed_sessions)


def classify_fact_type(question: str, answer: str, session_ids: List[int]) -> str:
    """Bucket QA for reporting: early / recent / temporal / preference_event."""
    q = question or ""
    a = answer or ""
    if _DATE_HINT_RE.search(q) or _DATE_HINT_RE.search(a):
        return "temporal_date"
    if _PREF_HINT_RE.search(q):
        return "preference_event"
    if session_ids and max(session_ids) <= 2:
        return "early_session"
    if session_ids and min(session_ids) >= 3:
        return "recent_session"
    return "preference_event"


def select_evidence_aligned_qa(
    sample: Dict[str, Any],
    *,
    max_closed_sessions: int,
    max_qa: int = 5,
) -> List[Dict[str, Any]]:
    """Keep QA whose evidence lies entirely inside seeded sessions; diversify fact types."""
    pairs = normalize_qa_pairs(sample)
    covered = [qa for qa in pairs if qa_evidence_covered(qa, max_closed_sessions)]
    if not covered:
        return []

    # Prefer diversity across fact types, then earlier evidence (harder for recency truncation).
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for qa in covered:
        by_type.setdefault(qa["fact_type"], []).append(qa)
    for bucket in by_type.values():
        bucket.sort(key=lambda item: (item.get("max_evidence_session") or 0, item["qa_index"]))

    order = ["early_session", "temporal_date", "preference_event", "recent_session"]
    selected: List[Dict[str, Any]] = []
    # Round-robin across types
    while len(selected) < max_qa:
        progressed = False
        for fact_type in order:
            bucket = by_type.get(fact_type) or []
            while bucket:
                candidate = bucket.pop(0)
                if candidate["qa_index"] in {row["qa_index"] for row in selected}:
                    continue
                selected.append(candidate)
                progressed = True
                break
            if len(selected) >= max_qa:
                break
        if not progressed:
            break

    if len(selected) < max_qa:
        remaining = [
            qa
            for qa in covered
            if qa["qa_index"] not in {row["qa_index"] for row in selected}
        ]
        remaining.sort(key=lambda item: (item.get("max_evidence_session") or 0, item["qa_index"]))
        selected.extend(remaining[: max_qa - len(selected)])

    return selected[:max_qa]


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = re.findall(r"\w+", (prediction or "").lower())
    gold_tokens = re.findall(r"\w+", (gold or "").lower())
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    pred_counts: Dict[str, int] = {}
    gold_counts: Dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    overlap = 0
    for token, count in gold_counts.items():
        overlap += min(count, pred_counts.get(token, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def exact_match(prediction: str, gold: str) -> bool:
    return (prediction or "").strip().lower() == (gold or "").strip().lower()


def get_subset_summary(
    samples: Optional[List[Dict[str, Any]]] = None,
    target_n: int = 20,
) -> Dict[str, Any]:
    loaded = samples if samples is not None else load_locomo_samples()
    subset = select_stratified_subset(loaded, target_n=target_n)
    return {
        "source": "snap-research/LoCoMo locomo10.json",
        "available_samples": len(loaded),
        "subset_size": len(subset),
        "target_n": target_n,
        "samples": [sample_stats(item) for item in subset],
    }
