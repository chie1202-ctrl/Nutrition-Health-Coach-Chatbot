#!/usr/bin/env python3
"""
Benchmark DeepSeek local models for coaching chat: latency vs reply quality.

Usage (from backend/):
  python eval/benchmark_llm_models.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import logic  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

BENCHMARK_PROMPTS = [
    {
        "id": "P01_goal_recall",
        "message": "What was my goal again, and what food should I avoid?",
        "profile": {
            "name": "Alex",
            "gender": "female",
            "goal": "lose_weight",
            "allergies": ["shellfish"],
            "diet_preference": "high_protein",
        },
        "metrics": {"weight_kg": 68, "bmi": 24.2, "bmi_label": "Normal", "ree": 1450},
        "expected_keywords": ["shellfish"],
    },
    {
        "id": "P02_breakfast_advice",
        "message": "Last time we discussed my breakfast drink problem. What should I try now?",
        "profile": {
            "name": "Alex",
            "gender": "female",
            "goal": "maintain_weight",
            "allergies": [],
            "self_description": "milk at breakfast causes bloating",
        },
        "metrics": {"weight_kg": 60, "bmi": 22.0, "bmi_label": "Normal", "ree": 1380},
        "expected_keywords": ["milk", "lactose", "soy"],
    },
    {
        "id": "P03_rag_vegetables",
        "message": "What vegetables do dietary guidelines recommend eating more of?",
        "profile": {
            "name": "Alex",
            "gender": "female",
            "goal": "healthy_eating",
            "allergies": [],
        },
        "metrics": {"weight_kg": 62, "bmi": 22.5, "bmi_label": "Normal", "ree": 1400},
        "expected_keywords": ["vegetable", "vegetables", "fiber", "nutrient"],
    },
]

CONFIGS = [
    {"config_id": "r1_7b_reasoning_default", "model": "deepseek-r1:7b", "num_predict": 384, "temperature": 0.3, "reasoning": None},
    {"config_id": "r1_7b_no_reasoning_np384", "model": "deepseek-r1:7b", "num_predict": 384, "temperature": 0.3, "reasoning": False},
    {"config_id": "r1_7b_no_reasoning_np256", "model": "deepseek-r1:7b", "num_predict": 256, "temperature": 0.3, "reasoning": False},
    {"config_id": "r1_8b_no_reasoning_np384", "model": "deepseek-r1:8b", "num_predict": 384, "temperature": 0.3, "reasoning": False},
    {"config_id": "r1_8b_no_reasoning_np512", "model": "deepseek-r1:8b", "num_predict": 512, "temperature": 0.3, "reasoning": False},
]


def ollama_has_model(name: str) -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as resp:
            data = json.loads(resp.read().decode())
        return any(name in m.get("name", "") for m in data.get("models", []))
    except Exception:
        return False


def keyword_score(reply: str, keywords: List[str]) -> float:
    low = logic.strip_think_tags(reply).lower()
    if not keywords:
        return 1.0
    hits = sum(1 for kw in keywords if kw.lower() in low)
    return hits / len(keywords)


def build_prompt(case: Dict[str, Any]) -> str:
    memory_context = {
        "memory_text": (
            "[Long-term Summary]\nUser wants to lose 5 kg in three months; prefers high-protein lunches; allergic to shellfish.\n\n"
            "[Recent Session Summary]\nDiscussed breakfast bloating after milk; suggested lactose-free or soy alternatives."
        )
    }
    return logic.build_coach_prompt(
        user=case["profile"],
        latest=case.get("metrics"),
        message=case["message"],
        rag_context="Eat a variety of vegetables, whole grains, lean proteins, and limit added sugars per dietary guidelines.",
        memory_context=memory_context,
    )


def run_config(config: Dict[str, Any], cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    model = config["model"]
    if not ollama_has_model(model):
        return {"config_id": config["config_id"], "skipped": True, "reason": f"model {model} not installed"}

    llm = logic.create_ollama_llm(
        model=model,
        num_predict=config.get("num_predict"),
        temperature=config.get("temperature"),
        reasoning=config.get("reasoning"),
    )
    runs: List[Dict[str, Any]] = []

    for case in cases:
        prompt = build_prompt(case)
        t0 = time.time()
        error = None
        raw = ""
        try:
            raw = llm.invoke(prompt) or ""
        except Exception as exc:
            error = str(exc)
        latency_ms = round((time.time() - t0) * 1000)
        visible = logic.strip_think_tags(raw)
        runs.append({
            "prompt_id": case["id"],
            "latency_ms": latency_ms,
            "error": error,
            "raw_len": len(raw),
            "visible_len": len(visible),
            "keyword_score": keyword_score(visible, case.get("expected_keywords", [])),
            "reply_preview": visible[:220],
        })

    latencies = [r["latency_ms"] for r in runs if not r["error"]]
    scores = [r["keyword_score"] for r in runs if not r["error"]]
    min_visible = min((r["visible_len"] for r in runs if not r["error"]), default=0)
    return {
        "config_id": config["config_id"],
        "model": model,
        "num_predict": config.get("num_predict"),
        "temperature": config.get("temperature"),
        "reasoning": config.get("reasoning"),
        "skipped": False,
        "runs": runs,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "avg_keyword_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "min_visible_len": min_visible,
        "error_count": sum(1 for r in runs if r["error"]),
    }


def score_config(result: Dict[str, Any]) -> Optional[float]:
    if result.get("skipped") or result.get("error_count", 0) > 0:
        return None
    if not result.get("avg_latency_ms") or result.get("min_visible_len", 0) < 40:
        return None
    latency = result["avg_latency_ms"]
    quality = result["avg_keyword_score"]
    latency_penalty = max(0.0, (latency - 12000) / 45000)
    return round(quality * 2.0 - latency_penalty, 4)


def main() -> int:
    if not logic.check_ollama_reachable():
        print("Ollama not reachable. Start with: ollama serve", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: Dict[str, Any] = {
        "timestamp": timestamp,
        "ollama_reachable": True,
        "configs_tested": [],
        "ranking": [],
    }

    for config in CONFIGS:
        print(f"Running {config['config_id']}...", flush=True)
        outcome = run_config(config, BENCHMARK_PROMPTS)
        results["configs_tested"].append(outcome)
        composite = score_config(outcome)
        if composite is not None:
            results["ranking"].append({
                "config_id": outcome["config_id"],
                "model": outcome["model"],
                "num_predict": outcome.get("num_predict"),
                "reasoning": outcome.get("reasoning"),
                "avg_latency_ms": outcome["avg_latency_ms"],
                "avg_keyword_score": outcome["avg_keyword_score"],
                "min_visible_len": outcome["min_visible_len"],
                "composite_score": composite,
            })

    results["ranking"].sort(key=lambda x: x["composite_score"], reverse=True)
    results["recommended"] = results["ranking"][0] if results["ranking"] else None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"llm_benchmark_{timestamp}.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
