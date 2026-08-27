#!/usr/bin/env python3
"""AI-graded semantic cross-session memory evaluation.

This runner strengthens the legacy keyword-only memory ablation by:

1. Expanding scripted nutrition-coaching scenarios.
2. Repeating each memory condition.
3. Grading final replies with a fixed AI-evaluator rubric.

RAG is disabled so results isolate cross-session memory behaviour.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import logic  # noqa: E402
from eval_db import add_eval_db_arguments, eval_db_metadata, setup_eval_database  # noqa: E402

SCRIPTS_PATH = os.path.join(EVAL_DIR, "memory_semantic_scripts.json")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
DEFAULT_MODES = ["M0", "M1", "M2", "M3"]
EVAL_REPLY_ENGLISH_SUFFIX = "\n\nReply in English only."
DEFAULT_OPENAI_JUDGE_MODEL = "gpt-4o-mini-2024-07-18"


def _load_scripts(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Semantic memory scripts must be a JSON list")
    return data


def _eval_user_prompt(prompt: str) -> str:
    if os.getenv("EVAL_REPLY_ENGLISH", "true").strip().lower() in ("false", "0", "no"):
        return prompt
    return f"{prompt}{EVAL_REPLY_ENGLISH_SUFFIX}"


def _reset_eval_user(name: str) -> int:
    for user in logic.get_all_users():
        if user.get("name") == name:
            logic.delete_user(int(user["user_id"]))
    return logic.create_user_profile(
        name=name,
        gender="female",
        birth_date="1990-01-01",
        height_cm=165.0,
        initial_weight_kg=68.0,
        goal="general_wellbeing",
        diet_preference="balanced",
        allergies=[],
        medical_conditions=[],
        food_dislikes=[],
    )


def _keyword_hits(text: str, keywords: List[str]) -> Dict[str, bool]:
    low = (text or "").lower()
    return {keyword: keyword.lower() in low for keyword in keywords}


def _keyword_pass_rate(checks: Dict[str, bool]) -> float:
    if not checks:
        return 0.0
    return round(sum(1 for ok in checks.values() if ok) / len(checks), 3)


def _summary_fidelity(summary: str, expected_facts: List[str]) -> Dict[str, Any]:
    low = (summary or "").lower()
    hits = [fact for fact in expected_facts if fact.lower() in low]
    total = len(expected_facts)
    return {
        "recall": round(len(hits) / total, 3) if total else 0.0,
        "hits": hits,
        "missed": [fact for fact in expected_facts if fact.lower() not in low],
    }


def _seed_session_turns(user_id: int, turns: List[Dict[str, str]], mode: str) -> Dict[str, Any]:
    session_id, _ = logic.resolve_session(user_id, force_new=True)
    for turn in turns:
        logic.save_chat(user_id, turn["role"], turn["content"], session_id=session_id)
    logic.close_session(session_id, user_id, trigger_summarization=False)

    summary: Optional[Dict[str, Any]] = None
    if mode in {"M1", "M2"} and len(turns) >= 2:
        summary = logic.summarize_session(session_id, user_id)
        # M1 is evaluated as cumulative-summary-only memory. Calling rollup here
        # ensures M1 has a fair chance to build cumulative memory after enough
        # closed sessions, while M2 still adds recent session summaries.
        logic.maybe_rollup_memory(user_id)

    return {"session_id": session_id, "summary": summary}


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = logic.strip_think_tags(text or "").strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("Evaluator did not return a JSON object")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Evaluator JSON was not an object")
    return parsed


def _clamp_score(value: Any) -> Optional[int]:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, min(2, score))


def _normalise_judge_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    memory_correctness = _clamp_score(raw.get("memory_correctness"))
    contextual_use = _clamp_score(raw.get("contextual_use"))
    unsupported_user_facts = _clamp_score(raw.get("unsupported_user_facts"))
    scores = [memory_correctness, contextual_use, unsupported_user_facts]
    semantic_score = None
    if all(score is not None for score in scores):
        semantic_score = round(sum(int(score) for score in scores) / 6, 3)

    scenario_specific_pass = raw.get("scenario_specific_pass")
    if isinstance(scenario_specific_pass, str):
        scenario_specific_pass = scenario_specific_pass.strip().lower() in ("true", "yes", "pass", "1")
    elif scenario_specific_pass is not None:
        scenario_specific_pass = bool(scenario_specific_pass)

    return {
        "memory_correctness": memory_correctness,
        "contextual_use": contextual_use,
        "unsupported_user_facts": unsupported_user_facts,
        "semantic_memory_score": semantic_score,
        "scenario_specific_pass": scenario_specific_pass,
        "brief_reason": str(raw.get("brief_reason") or "").strip()[:600],
    }


def _judge_prompt(script: Dict[str, Any], final_prompt: str, reply: str) -> str:
    ground = script.get("ground_truth") or {}
    payload = {
        "script_id": script.get("script_id"),
        "description": script.get("description"),
        "ground_truth": ground,
        "final_user_prompt": final_prompt,
        "model_response": reply,
    }
    return f"""You are an impartial evaluator for a cross-session memory experiment.
Evaluate only whether the model response correctly uses the provided ground-truth user history.
Do not reward general nutrition advice unless it uses the relevant remembered context.
Do not assume facts that are not in the ground truth.

Rubric:
- memory_correctness: 0 = relevant user context missing or incorrect; 1 = partially correct; 2 = correctly recalls the relevant context.
- contextual_use: 0 = memory not used or used incorrectly; 1 = relevant but weak/incomplete use; 2 = correctly applies memory to the response.
- unsupported_user_facts: 0 = introduces incorrect/unsupported user-specific facts; 1 = minor ambiguous inference; 2 = no unsupported user-specific claims.
- scenario_specific_pass: true only if the response satisfies ground_truth.scenario_specific_check and expected_behaviour.

Return JSON only with this schema:
{{
  "memory_correctness": 0,
  "contextual_use": 0,
  "unsupported_user_facts": 0,
  "scenario_specific_pass": false,
  "brief_reason": "one concise reason"
}}

Evaluation item:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


JUDGE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "memory_correctness": {"type": "integer", "minimum": 0, "maximum": 2},
        "contextual_use": {"type": "integer", "minimum": 0, "maximum": 2},
        "unsupported_user_facts": {"type": "integer", "minimum": 0, "maximum": 2},
        "scenario_specific_pass": {"type": "boolean"},
        "brief_reason": {"type": "string"},
    },
    "required": [
        "memory_correctness",
        "contextual_use",
        "unsupported_user_facts",
        "scenario_specific_pass",
        "brief_reason",
    ],
}


def _responses_output_text(response: Dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct

    chunks: List[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def openai_grade_reply(script: Dict[str, Any], final_prompt: str, reply: str, judge_model: str = "") -> Dict[str, Any]:
    started = time.perf_counter()
    raw_excerpt = ""
    try:
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        model = judge_model or os.getenv("OPENAI_JUDGE_MODEL") or DEFAULT_OPENAI_JUDGE_MODEL
        prompt = _judge_prompt(script, final_prompt, reply)
        payload = {
            "model": model,
            "input": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict, impartial evaluator. Return only the requested structured JSON. "
                        "You are not a nutrition expert; grade only memory correctness and response use against the supplied ground truth."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "memory_semantic_judge",
                    "schema": JUDGE_JSON_SCHEMA,
                    "strict": True,
                }
            },
            "max_output_tokens": int(os.getenv("OPENAI_JUDGE_MAX_OUTPUT_TOKENS", "600")),
        }
        timeout = float(os.getenv("OPENAI_JUDGE_TIMEOUT", "60"))
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        if response.status_code >= 400:
            raise ValueError(f"OpenAI judge request failed {response.status_code}: {response.text[:500]}")
        data = response.json()
        output_text = _responses_output_text(data)
        raw_excerpt = output_text[:600]
        parsed = _extract_json_object(output_text)
        normalised = _normalise_judge_result(parsed)
        normalised["judge_error"] = None
        normalised["judge_model"] = model
        normalised["judge_provider"] = "openai"
    except Exception as exc:  # noqa: BLE001
        normalised = {
            "memory_correctness": None,
            "contextual_use": None,
            "unsupported_user_facts": None,
            "semantic_memory_score": None,
            "scenario_specific_pass": None,
            "brief_reason": "",
            "judge_error": str(exc),
            "judge_model": judge_model or os.getenv("OPENAI_JUDGE_MODEL") or DEFAULT_OPENAI_JUDGE_MODEL,
            "judge_provider": "openai",
        }
    normalised["raw_judge_excerpt"] = raw_excerpt
    normalised["judge_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return normalised


def ollama_grade_reply(script: Dict[str, Any], final_prompt: str, reply: str, judge_model: str = "") -> Dict[str, Any]:
    started = time.perf_counter()
    prompt = _judge_prompt(script, final_prompt, reply)
    raw_excerpt = ""
    try:
        llm = logic.create_ollama_llm(
            model=judge_model or os.getenv("EVAL_JUDGE_MODEL") or logic.get_ollama_chat_model_name(),
            num_predict=int(os.getenv("EVAL_JUDGE_NUM_PREDICT", "1536")),
            temperature=float(os.getenv("EVAL_JUDGE_TEMPERATURE", "0")),
            reasoning=False,
        )
        parsed = None
        last_error: Optional[Exception] = None
        for attempt in range(1, 3):
            use_prompt = prompt
            if attempt == 2:
                use_prompt = (
                    f"{prompt}\n\n"
                    "IMPORTANT: Your previous answer was not valid JSON. "
                    "Return exactly one JSON object and no prose, no markdown, no explanation."
                )
            raw = llm.invoke(use_prompt)
            raw_text = raw if isinstance(raw, str) else str(raw or "")
            stripped = logic.strip_think_tags(raw_text).strip()
            raw_excerpt = (stripped or raw_text).strip()[:600]
            try:
                parsed = _extract_json_object(raw_text)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if parsed is None:
            raise ValueError(str(last_error or "Evaluator did not return valid JSON"))
        normalised = _normalise_judge_result(parsed)
        normalised["judge_error"] = None
        normalised["judge_model"] = judge_model or os.getenv("EVAL_JUDGE_MODEL") or logic.get_ollama_chat_model_name()
        normalised["judge_provider"] = "ollama"
    except Exception as exc:  # noqa: BLE001
        normalised = {
            "memory_correctness": None,
            "contextual_use": None,
            "unsupported_user_facts": None,
            "semantic_memory_score": None,
            "scenario_specific_pass": None,
            "brief_reason": "",
            "judge_error": str(exc),
            "judge_model": judge_model or os.getenv("EVAL_JUDGE_MODEL") or logic.get_ollama_chat_model_name(),
            "judge_provider": "ollama",
        }
    normalised["raw_judge_excerpt"] = raw_excerpt
    normalised["judge_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return normalised


def ai_grade_reply(
    script: Dict[str, Any],
    final_prompt: str,
    reply: str,
    *,
    judge_provider: str = "openai",
    judge_model: str = "",
) -> Dict[str, Any]:
    provider = (judge_provider or "openai").strip().lower()
    if provider == "openai":
        return openai_grade_reply(script, final_prompt, reply, judge_model=judge_model)
    if provider == "ollama":
        return ollama_grade_reply(script, final_prompt, reply, judge_model=judge_model)
    raise ValueError(f"Unsupported judge provider: {judge_provider}")


def run_one(
    script: Dict[str, Any],
    mode: str,
    repeat: int,
    judge_provider: str = "openai",
    judge_model: str = "",
    skip_ai_grading: bool = False,
) -> Dict[str, Any]:
    user_label = f"SemanticEval_{script['script_id']}_{mode}_r{repeat}"
    user_id = _reset_eval_user(user_label)
    logic.ensure_user_memory_state(user_id)

    session_records: List[Dict[str, Any]] = []
    summary_records: List[Dict[str, Any]] = []
    for index, session in enumerate(script["sessions"][:-1]):
        seeded = _seed_session_turns(user_id, session.get("turns", []), mode)
        session_records.append(
            {
                "session_label": session.get("session_label", f"session_{index + 1}"),
                "session_id": seeded["session_id"],
            }
        )
        if seeded.get("summary") and session.get("expected_key_facts"):
            summary_records.append(
                {
                    "session_label": session.get("session_label", f"session_{index + 1}"),
                    "fidelity": _summary_fidelity(seeded["summary"].get("content", ""), session["expected_key_facts"]),
                }
            )

    final_session = script["sessions"][-1]
    logic.resolve_session(user_id, force_new=True)
    final_prompt = _eval_user_prompt(final_session["user_prompt"])
    started = time.perf_counter()
    response = logic.process_chat_message(
        user_id,
        final_prompt,
        rag_store=None,
        force_new_session=False,
        memory_mode=mode,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    reply = response.get("reply") or ""
    keyword_checks = _keyword_hits(reply, final_session.get("expected_keywords", []))

    judge = None if skip_ai_grading else ai_grade_reply(
        script,
        final_prompt,
        reply,
        judge_provider=judge_provider,
        judge_model=judge_model,
    )

    return {
        "script_id": script["script_id"],
        "description": script.get("description", ""),
        "memory_mode": mode,
        "repeat": repeat,
        "user_id": user_id,
        "final_prompt": final_prompt,
        "reply": reply,
        "reply_excerpt": reply[:300],
        "ground_truth": script.get("ground_truth") or {},
        "memory_used": response.get("memory_used", {}),
        "keyword_checks": keyword_checks,
        "keyword_pass_rate": _keyword_pass_rate(keyword_checks),
        "ai_judge": judge,
        "latency_ms": latency_ms,
        "session_records": session_records,
        "summary_fidelity": summary_records,
    }


def _mean(values: List[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    return round(statistics.mean(clean), 4) if clean else None


def _stdev(values: List[float]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    return round(statistics.stdev(clean), 4) if len(clean) >= 2 else None


def aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    by_scenario_mode: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["memory_mode"], []).append(row)
        by_scenario_mode.setdefault((row["script_id"], row["memory_mode"]), []).append(row)

    mode_summary = []
    for mode, items in sorted(by_mode.items()):
        semantic_scores = [
            (item.get("ai_judge") or {}).get("semantic_memory_score")
            for item in items
            if item.get("ai_judge") is not None
        ]
        specific = [
            1.0 if (item.get("ai_judge") or {}).get("scenario_specific_pass") is True else 0.0
            for item in items
            if (item.get("ai_judge") or {}).get("scenario_specific_pass") is not None
        ]
        tokens = [
            (item.get("memory_used") or {}).get("estimated_memory_tokens")
            for item in items
            if (item.get("memory_used") or {}).get("estimated_memory_tokens") is not None
        ]
        unsupported = [
            (item.get("ai_judge") or {}).get("unsupported_user_facts")
            for item in items
            if (item.get("ai_judge") or {}).get("unsupported_user_facts") is not None
        ]
        mode_summary.append(
            {
                "memory_mode": mode,
                "runs": len(items),
                "mean_semantic_memory_score": _mean(semantic_scores),
                "stdev_semantic_memory_score": _stdev(semantic_scores),
                "mean_keyword_pass_rate": _mean([item.get("keyword_pass_rate") for item in items]),
                "scenario_specific_pass_rate": _mean(specific),
                "mean_unsupported_user_facts_score": _mean(unsupported),
                "mean_injected_tokens": _mean(tokens),
                "mean_latency_ms": _mean([item.get("latency_ms") for item in items]),
                "judge_errors": sum(1 for item in items if (item.get("ai_judge") or {}).get("judge_error")),
            }
        )

    scenario_summary = []
    for (script_id, mode), items in sorted(by_scenario_mode.items()):
        semantic_scores = [
            (item.get("ai_judge") or {}).get("semantic_memory_score")
            for item in items
            if item.get("ai_judge") is not None
        ]
        scenario_summary.append(
            {
                "script_id": script_id,
                "memory_mode": mode,
                "runs": len(items),
                "mean_semantic_memory_score": _mean(semantic_scores),
                "mean_keyword_pass_rate": _mean([item.get("keyword_pass_rate") for item in items]),
                "mean_injected_tokens": _mean([
                    (item.get("memory_used") or {}).get("estimated_memory_tokens")
                    for item in items
                    if (item.get("memory_used") or {}).get("estimated_memory_tokens") is not None
                ]),
            }
        )

    return {"by_mode": mode_summary, "by_scenario_mode": scenario_summary}


def write_outputs(rows: List[Dict[str, Any]], args: argparse.Namespace, scripts_path: str) -> Dict[str, str]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(RESULTS_DIR, f"memory_semantic_eval_{stamp}.json")
    csv_path = os.path.join(RESULTS_DIR, f"memory_semantic_eval_{stamp}.csv")
    summary_csv_path = os.path.join(RESULTS_DIR, f"memory_semantic_eval_summary_{stamp}.csv")

    aggregate = aggregate_rows(rows)
    payload = {
        "eval": "memory_semantic_eval",
        "created_at": stamp,
        "scripts_path": scripts_path,
        "config": {
            "modes": args.modes,
            "repeats": args.repeats,
            "max_scenarios": args.max_scenarios,
            "rag": "off",
            "judge_provider": args.judge_provider,
            "judge_model": (
                args.judge_model
                or (os.getenv("OPENAI_JUDGE_MODEL") if args.judge_provider == "openai" else os.getenv("EVAL_JUDGE_MODEL"))
                or (DEFAULT_OPENAI_JUDGE_MODEL if args.judge_provider == "openai" else logic.get_ollama_chat_model_name())
            ),
            "ollama_model": logic.get_ollama_chat_model_name(),
            "ollama_reasoning": os.getenv("OLLAMA_REASONING"),
            "ollama_temperature": os.getenv("OLLAMA_TEMPERATURE"),
            "summary_model": logic.get_summary_model_name(),
            "summary_temperature": os.getenv("SUMMARY_TEMPERATURE"),
            "memory_budget_enabled": logic.memory_budget_enabled(),
            "database": eval_db_metadata(),
        },
        "aggregate": aggregate,
        "rows": rows,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "script_id",
                "memory_mode",
                "repeat",
                "keyword_pass_rate",
                "semantic_memory_score",
                "memory_correctness",
                "contextual_use",
                "unsupported_user_facts",
                "scenario_specific_pass",
                "estimated_memory_tokens",
                "latency_ms",
                "judge_error",
                "reply_excerpt",
                "judge_reason",
            ],
        )
        writer.writeheader()
        for row in rows:
            judge = row.get("ai_judge") or {}
            memory_used = row.get("memory_used") or {}
            writer.writerow(
                {
                    "script_id": row["script_id"],
                    "memory_mode": row["memory_mode"],
                    "repeat": row["repeat"],
                    "keyword_pass_rate": row["keyword_pass_rate"],
                    "semantic_memory_score": judge.get("semantic_memory_score"),
                    "memory_correctness": judge.get("memory_correctness"),
                    "contextual_use": judge.get("contextual_use"),
                    "unsupported_user_facts": judge.get("unsupported_user_facts"),
                    "scenario_specific_pass": judge.get("scenario_specific_pass"),
                    "estimated_memory_tokens": memory_used.get("estimated_memory_tokens"),
                    "latency_ms": row["latency_ms"],
                    "judge_error": judge.get("judge_error"),
                    "reply_excerpt": row["reply_excerpt"],
                    "judge_reason": judge.get("brief_reason"),
                }
            )

    with open(summary_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate["by_mode"][0].keys()) if aggregate["by_mode"] else [])
        if aggregate["by_mode"]:
            writer.writeheader()
            writer.writerows(aggregate["by_mode"])

    return {"json": json_path, "csv": csv_path, "summary_csv": summary_csv_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-graded semantic memory ablation eval")
    add_eval_db_arguments(parser)
    parser.add_argument("--scripts", default=SCRIPTS_PATH, help="Path to semantic memory scripts JSON")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help="Comma-separated memory modes")
    parser.add_argument("--repeats", type=int, default=3, help="Repeated runs per scenario/mode")
    parser.add_argument("--max-scenarios", type=int, default=0, help="Limit scenarios for smoke tests")
    parser.add_argument(
        "--judge-provider",
        choices=["openai", "ollama"],
        default="openai",
        help="AI evaluator provider. Use openai for formal semantic grading; ollama is for local smoke tests.",
    )
    parser.add_argument("--judge-model", default="", help="Optional evaluator model")
    parser.add_argument("--skip-ai-grading", action="store_true", help="Generate replies without AI grading")
    args = parser.parse_args()

    load_dotenv(os.path.join(BACKEND_DIR, ".env"))

    setup_eval_database(args)
    scripts = _load_scripts(args.scripts)
    if args.max_scenarios:
        scripts = scripts[: max(0, args.max_scenarios)]

    modes = [mode.strip().upper() for mode in args.modes.split(",") if mode.strip()]
    modes = [mode for mode in modes if mode in logic.MEMORY_MODES]
    if not modes:
        raise ValueError("No valid memory modes selected")

    rows: List[Dict[str, Any]] = []
    total = len(scripts) * len(modes) * max(1, args.repeats)
    completed = 0
    for script in scripts:
        for mode in modes:
            for repeat in range(1, max(1, args.repeats) + 1):
                completed += 1
                print(f"[{completed}/{total}] {script['script_id']} {mode} repeat={repeat}", flush=True)
                rows.append(
                    run_one(
                        script,
                        mode,
                        repeat,
                        judge_provider=args.judge_provider,
                        judge_model=args.judge_model,
                        skip_ai_grading=args.skip_ai_grading,
                    )
                )

    paths = write_outputs(rows, args, args.scripts)
    print(f"Wrote JSON: {paths['json']}", flush=True)
    print(f"Wrote CSV: {paths['csv']}", flush=True)
    print(f"Wrote summary CSV: {paths['summary_csv']}", flush=True)
    print(json.dumps(aggregate_rows(rows)["by_mode"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
