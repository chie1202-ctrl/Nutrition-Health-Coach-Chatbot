import json
import os
from typing import Any, List, Literal, Optional

from dotenv import load_dotenv

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_FILE = os.path.join(_BACKEND_DIR, ".env")
_ENV_EXAMPLE = os.path.join(_BACKEND_DIR, ".env.example")
if os.path.isfile(_ENV_FILE):
    load_dotenv(_ENV_FILE)
else:
    load_dotenv(_ENV_EXAMPLE)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import logic

app = FastAPI(title="NutriCoachAI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserPayload(BaseModel):
    name: str = Field(..., min_length=1)
    gender: Literal["male", "female"]
    birth_date: str = Field(..., min_length=8)
    height_cm: float = Field(..., gt=0)
    weight_kg: Optional[float] = None
    goal: str = ""
    activity_level: str = ""
    diet_preference: str = ""
    budget_level: str = ""
    medical_conditions: List[str] = []
    allergies: List[str] = []
    food_dislikes: List[str] = []
    target_weight: str = ""
    target_timeline: str = ""
    self_description: str = ""
    coach_notes: str = ""


class WeightPayload(BaseModel):
    user_id: int
    weight_kg: float = Field(..., gt=0)
    recorded_at: str | None = None
    note: str | None = None


class ChatPayload(BaseModel):
    user_id: int
    message: str = Field(..., min_length=1)
    force_new_session: bool = False


class RegenerateSummaryPayload(BaseModel):
    session_id: int


@app.on_event("startup")
def startup_event():
    logic.init_db()
    app.state.rag_store = logic.initialize_rag()


@app.get("/health")
def health_check():
    runtime = logic.get_runtime_health()
    return {
        "status": "ok",
        "db_path": logic.DB_PATH,
        "pdf_dir": logic.PDF_DIR,
        "ollama_model": runtime["ollama_model"],
        "ollama_reachable": runtime["ollama_reachable"],
        "llm_deps_available": runtime["llm_deps_available"],
        "rag_ready": app.state.rag_store is not None,
        "memory_feature_enabled": True,
        "session_idle_timeout_minutes": logic.session_idle_timeout_minutes(),
        "summary_model": runtime["summary_model"],
        "memory_mode": runtime["memory_mode"],
    }


@app.get("/users")
def list_users():
    return logic.get_all_users()


@app.post("/users")
def create_user(payload: UserPayload):
    user_id = logic.create_user_profile(
        payload.name,
        payload.gender,
        payload.birth_date,
        payload.height_cm,
        payload.weight_kg,
        goal=payload.goal,
        activity_level=payload.activity_level,
        diet_preference=payload.diet_preference,
        budget_level=payload.budget_level,
        medical_conditions=payload.medical_conditions,
        allergies=payload.allergies,
        food_dislikes=payload.food_dislikes,
        target_weight=payload.target_weight,
        target_timeline=payload.target_timeline,
        self_description=payload.self_description,
        coach_notes=payload.coach_notes,
    )
    logic.ensure_user_memory_state(user_id)
    user = logic.get_user_profile(user_id)
    return {"status": "success", "user": user}


@app.get("/users/{user_id}")
def get_user_bundle(user_id: int):
    user = logic.get_user_profile(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    latest = logic.get_latest_metrics_bundle(user_id)
    series = logic.get_weight_history(user_id)
    meal_plan = logic.get_latest_meal_plan(user_id)
    metrics = {
        "weight_kg": latest["weight_kg"] if latest else None,
        "bmi": latest["bmi"] if latest else None,
        "ree": latest["ree"] if latest else None,
        "bmi_label": latest["bmi_label"] if latest else None,
        "series": series,
    }
    return {"user": user, "metrics": metrics, "meal_plan": meal_plan}


@app.put("/users/{user_id}")
def update_user(user_id: int, payload: UserPayload):
    if not logic.get_user_profile(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    logic.update_user_profile(
        user_id,
        payload.name,
        payload.gender,
        payload.birth_date,
        payload.height_cm,
        goal=payload.goal,
        activity_level=payload.activity_level,
        diet_preference=payload.diet_preference,
        budget_level=payload.budget_level,
        medical_conditions=payload.medical_conditions,
        allergies=payload.allergies,
        food_dislikes=payload.food_dislikes,
        target_weight=payload.target_weight,
        target_timeline=payload.target_timeline,
        self_description=payload.self_description,
        coach_notes=payload.coach_notes,
    )
    if payload.weight_kg is not None:
        logic.upsert_weight_entry(user_id, payload.weight_kg)
    return {"status": "success", "user": logic.get_user_profile(user_id)}


@app.delete("/users/{user_id}")
def remove_user(user_id: int):
    if not logic.get_user_profile(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    logic.delete_user(user_id)
    return {"status": "success"}


@app.get("/users/{user_id}/metrics")
def get_metrics(user_id: int):
    if not logic.get_user_profile(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    latest = logic.get_latest_metrics_bundle(user_id)
    history = logic.get_weight_history(user_id)
    if not latest:
        return {"weight_kg": None, "bmi": None, "ree": None, "bmi_label": None, "series": history}
    return {
        "weight_kg": latest["weight_kg"],
        "bmi": latest["bmi"],
        "ree": latest["ree"],
        "bmi_label": latest["bmi_label"],
        "series": history,
    }


@app.get("/users/{user_id}/weights")
def list_weights(user_id: int):
    if not logic.get_user_profile(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return logic.get_weight_history(user_id)


@app.post("/users/{user_id}/weight")
def add_or_update_weight(user_id: int, payload: WeightPayload):
    if payload.user_id != user_id:
        raise HTTPException(status_code=400, detail="Payload user_id does not match URL user_id")
    if not logic.get_user_profile(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return logic.upsert_weight_entry(user_id, payload.weight_kg, payload.recorded_at, payload.note)


@app.put("/weights/{metric_id}")
def update_weight(metric_id: int, payload: WeightPayload):
    if not logic.get_user_profile(payload.user_id):
        raise HTTPException(status_code=404, detail="User not found")
    recorded_at = payload.recorded_at or logic.utc_now_str()
    return logic.update_weight_record(metric_id, payload.user_id, payload.weight_kg, recorded_at, payload.note)


@app.delete("/weights/{metric_id}")
def remove_weight(metric_id: int):
    logic.delete_weight_record(metric_id)
    return {"status": "success"}


@app.get("/users/{user_id}/chat")
def get_history(user_id: int):
    if not logic.get_user_profile(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return logic.get_chat_history(user_id)


@app.get("/users/{user_id}/memory")
def get_memory(user_id: int):
    if not logic.get_user_profile(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return logic.get_user_memory_bundle(user_id)


@app.get("/users/{user_id}/summaries")
def get_summaries(user_id: int, limit: int = 10):
    if not logic.get_user_profile(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return logic.list_user_summaries(user_id, limit=limit)


@app.get("/users/{user_id}/sessions")
def get_sessions(user_id: int, limit: int = 20):
    if not logic.get_user_profile(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return logic.list_user_sessions(user_id, limit=limit)


@app.post("/users/{user_id}/sessions/close")
def close_session(user_id: int):
    if not logic.get_user_profile(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return logic.close_user_session(user_id)


@app.post("/users/{user_id}/summaries/regenerate")
def regenerate_summary(user_id: int, payload: RegenerateSummaryPayload):
    if not logic.get_user_profile(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    result = logic.regenerate_session_summary(payload.session_id, user_id)
    if not result:
        raise HTTPException(status_code=400, detail="Unable to regenerate summary for this session")
    return {"status": "success", "summary": result}


@app.get("/users/{user_id}/meal-plan")
def get_meal_plan(user_id: int):
    if not logic.get_user_profile(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return logic.get_latest_meal_plan(user_id) or {"plan": {"summary": "No meal plan generated yet.", "days": []}}


@app.post("/users/{user_id}/meal-plan")
def create_meal_plan(user_id: int):
    user = logic.get_user_profile(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    latest = logic.get_latest_metrics_bundle(user_id)
    try:
        plan, llm_degraded = logic.generate_meal_plan(user, latest, app.state.rag_store)
    except logic.OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    saved = logic.save_meal_plan(user_id, plan, user.get("goal", ""))
    runtime = logic.get_runtime_health()
    validation = logic.validate_meal_plan(plan, user)
    return {
        **saved,
        "ollama_reachable": runtime["ollama_reachable"],
        "llm_degraded": llm_degraded,
        "validation": validation,
    }


@app.post("/chat")
def chat(payload: ChatPayload):
    if not logic.get_user_profile(payload.user_id):
        raise HTTPException(status_code=404, detail="User not found")
    try:
        result = logic.process_chat_message(
            payload.user_id,
            payload.message,
            rag_store=app.state.rag_store,
            force_new_session=payload.force_new_session,
        )
    except logic.OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result


def _format_sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
def chat_stream(payload: ChatPayload):
    if not logic.get_user_profile(payload.user_id):
        raise HTTPException(status_code=404, detail="User not found")

    def event_generator():
        try:
            for item in logic.iter_chat_sse_events(
                payload.user_id,
                payload.message,
                rag_store=app.state.rag_store,
                force_new_session=payload.force_new_session,
            ):
                yield _format_sse(item["event"], item["data"])
        except logic.OllamaUnavailableError as exc:
            yield _format_sse("error", {"detail": str(exc)})
        except ValueError as exc:
            yield _format_sse("error", {"detail": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
