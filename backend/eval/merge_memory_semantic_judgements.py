#!/usr/bin/env python3
"""Merge copy-paste ChatGPT judgements back into semantic memory eval summaries."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _score(value: Any) -> Optional[int]:
    try:
        return max(0, min(2, int(value)))
    except (TypeError, ValueError):
        return None


def _mean(values: List[Any]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    return round(statistics.mean(clean), 4) if clean else None


def _stdev(values: List[Any]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    return round(statistics.stdev(clean), 4) if len(clean) >= 2 else None


def load_judgements(paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and isinstance(data.get("judgements"), list):
            rows = data["judgements"]
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError(f"{path} must contain a judgements array or a JSON list")
        for row in rows:
            item_id = str(row.get("item_id") or "").strip()
            if not item_id:
                continue
            by_id[item_id] = row
    return by_id


def merge(mapping_path: Path, judgement_paths: List[Path]) -> Dict[str, Any]:
    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, list):
        raise ValueError("mapping.json must contain a list")

    judgements = load_judgements(judgement_paths)
    rows: List[Dict[str, Any]] = []
    for item in mapping:
        item_id = item["item_id"]
        judge = judgements.get(item_id) or {}
        memory_correctness = _score(judge.get("memory_correctness"))
        contextual_use = _score(judge.get("contextual_use"))
        unsupported_user_facts = _score(judge.get("unsupported_user_facts"))
        semantic = None
        if None not in (memory_correctness, contextual_use, unsupported_user_facts):
            semantic = round((memory_correctness + contextual_use + unsupported_user_facts) / 6, 3)
        scenario_specific_pass = judge.get("scenario_specific_pass")
        if isinstance(scenario_specific_pass, str):
            scenario_specific_pass = scenario_specific_pass.strip().lower() in ("true", "yes", "pass", "1")
        elif scenario_specific_pass is not None:
            scenario_specific_pass = bool(scenario_specific_pass)

        rows.append(
            {
                **item,
                "memory_correctness": memory_correctness,
                "contextual_use": contextual_use,
                "unsupported_user_facts": unsupported_user_facts,
                "semantic_memory_score": semantic,
                "scenario_specific_pass": scenario_specific_pass,
                "brief_reason": str(judge.get("brief_reason") or ""),
                "missing_judgement": item_id not in judgements,
            }
        )

    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    by_scenario_mode: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["memory_mode"], []).append(row)
        by_scenario_mode.setdefault((row["script_id"], row["memory_mode"]), []).append(row)

    mode_summary = []
    for mode, items in sorted(by_mode.items()):
        specific = [
            1.0 if row.get("scenario_specific_pass") is True else 0.0
            for row in items
            if row.get("scenario_specific_pass") is not None
        ]
        mode_summary.append(
            {
                "memory_mode": mode,
                "runs": len(items),
                "mean_semantic_memory_score": _mean([row.get("semantic_memory_score") for row in items]),
                "stdev_semantic_memory_score": _stdev([row.get("semantic_memory_score") for row in items]),
                "mean_keyword_pass_rate": _mean([row.get("keyword_pass_rate") for row in items]),
                "scenario_specific_pass_rate": _mean(specific),
                "mean_unsupported_user_facts_score": _mean([row.get("unsupported_user_facts") for row in items]),
                "mean_injected_tokens": _mean([row.get("estimated_memory_tokens") for row in items]),
                "mean_latency_ms": _mean([row.get("latency_ms") for row in items]),
                "missing_judgements": sum(1 for row in items if row.get("missing_judgement")),
            }
        )

    scenario_summary = []
    for (script_id, mode), items in sorted(by_scenario_mode.items()):
        scenario_summary.append(
            {
                "script_id": script_id,
                "memory_mode": mode,
                "runs": len(items),
                "mean_semantic_memory_score": _mean([row.get("semantic_memory_score") for row in items]),
                "mean_keyword_pass_rate": _mean([row.get("keyword_pass_rate") for row in items]),
                "mean_injected_tokens": _mean([row.get("estimated_memory_tokens") for row in items]),
            }
        )

    return {"rows": rows, "by_mode": mode_summary, "by_scenario_mode": scenario_summary}


def write_outputs(result: Dict[str, Any], out_prefix: Path) -> Dict[str, str]:
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    summary_csv_path = out_prefix.with_name(out_prefix.name + "_summary.csv")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(result["rows"][0].keys()) if result["rows"] else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(result["rows"])

    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(result["by_mode"][0].keys()) if result["by_mode"] else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(result["by_mode"])

    return {"json": str(json_path), "csv": str(csv_path), "summary_csv": str(summary_csv_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ChatGPT semantic memory judgements")
    parser.add_argument("--mapping", required=True, help="mapping.json from make_memory_semantic_judge_pack.py")
    parser.add_argument("judgement_json", nargs="+", help="ChatGPT judgement JSON files")
    parser.add_argument("--out-prefix", default="", help="Output prefix without extension")
    args = parser.parse_args()

    mapping_path = Path(args.mapping).resolve()
    judgement_paths = [Path(path).resolve() for path in args.judgement_json]
    result = merge(mapping_path, judgement_paths)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_prefix = Path(args.out_prefix).resolve() if args.out_prefix else mapping_path.parent / f"merged_memory_semantic_judgements_{stamp}"
    paths = write_outputs(result, out_prefix)
    print(json.dumps({"outputs": paths, "by_mode": result["by_mode"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
