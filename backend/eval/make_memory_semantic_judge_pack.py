#!/usr/bin/env python3
"""Create copy-paste ChatGPT judging packs for semantic memory eval results.

Use this when avoiding OpenAI API calls:

1. Run ``run_memory_semantic_eval.py --skip-ai-grading``.
2. Run this script on the generated JSON.
3. Paste each ``chunk_XX.md`` into ChatGPT and save the JSON response.

The pack hides memory modes from the judge. A separate mapping file preserves
item_id -> script/mode/repeat for later aggregation.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


RUBRIC = """You are an impartial evaluator for a cross-session memory experiment.
Evaluate only whether the model response correctly uses the provided ground-truth user history.
Do not reward general nutrition advice unless it uses the relevant remembered context.
Do not assume facts that are not in the ground truth.

Rubric:
- memory_correctness: 0 = relevant user context missing or incorrect; 1 = partially correct; 2 = correctly recalls the relevant context.
- contextual_use: 0 = memory not used or used incorrectly; 1 = relevant but weak/incomplete use; 2 = correctly applies memory to the response.
- unsupported_user_facts: 0 = introduces incorrect/unsupported user-specific facts; 1 = minor ambiguous inference; 2 = no unsupported user-specific claims.
- scenario_specific_pass: true only if the response satisfies ground_truth.scenario_specific_check and expected_behaviour.

Return JSON only. Return one object with a "judgements" array. Each judgement must contain:
{
  "item_id": "the given item id",
  "memory_correctness": 0,
  "contextual_use": 0,
  "unsupported_user_facts": 0,
  "scenario_specific_pass": false,
  "brief_reason": "one concise reason"
}
"""


def load_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError("Expected a memory_semantic_eval JSON payload with a rows list")
    return payload


def make_items(rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    items: List[Dict[str, Any]] = []
    mapping: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        item_id = f"J{index:03d}"
        items.append(
            {
                "item_id": item_id,
                "script_id": row.get("script_id"),
                "description": row.get("description"),
                "ground_truth": row.get("ground_truth") or {},
                "final_user_prompt": row.get("final_prompt"),
                "model_response": row.get("reply"),
            }
        )
        mapping.append(
            {
                "item_id": item_id,
                "script_id": row.get("script_id"),
                "memory_mode": row.get("memory_mode"),
                "repeat": row.get("repeat"),
                "keyword_pass_rate": row.get("keyword_pass_rate"),
                "estimated_memory_tokens": (row.get("memory_used") or {}).get("estimated_memory_tokens"),
                "latency_ms": row.get("latency_ms"),
            }
        )
    return items, mapping


def write_pack(result_path: Path, out_dir: Path, chunk_size: int) -> Dict[str, Any]:
    payload = load_payload(result_path)
    items, mapping = make_items(payload["rows"])
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = out_dir / "mapping.json"
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "source_result": str(result_path),
        "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "chunk_size": chunk_size,
        "item_count": len(items),
        "mapping": str(mapping_path),
        "chunks": [],
    }

    for start in range(0, len(items), chunk_size):
        chunk_items = items[start : start + chunk_size]
        chunk_number = len(manifest["chunks"]) + 1
        chunk_path = out_dir / f"chunk_{chunk_number:02d}.md"
        body = (
            f"# Semantic Memory Judge Chunk {chunk_number:02d}\n\n"
            f"{RUBRIC}\n\n"
            "Evaluation items:\n\n"
            "```json\n"
            f"{json.dumps(chunk_items, ensure_ascii=False, indent=2)}\n"
            "```\n"
        )
        chunk_path.write_text(body, encoding="utf-8")
        manifest["chunks"].append({"path": str(chunk_path), "items": [item["item_id"] for item in chunk_items]})

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ChatGPT copy-paste judging packs")
    parser.add_argument("result_json", help="memory_semantic_eval_*.json from --skip-ai-grading")
    parser.add_argument("--chunk-size", type=int, default=8, help="Items per ChatGPT prompt chunk")
    parser.add_argument("--out-dir", default="", help="Output directory; default under backend/eval/results")
    args = parser.parse_args()

    result_path = Path(args.result_json).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else result_path.parent / f"memory_semantic_judge_pack_{stamp}"
    manifest = write_pack(result_path, out_dir, max(1, args.chunk_size))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
