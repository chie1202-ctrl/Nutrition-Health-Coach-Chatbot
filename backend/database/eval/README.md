# Isolated evaluation databases

Eval runners default to a SQLite file in this directory
(`health_coach_eval.db` or a path set via `EVAL_DB_PATH` / `--eval-db`).

This keeps LoCoMo / long-context / memory ablation seeds out of the
production `../health_coach.db` used by the NutriCoachAI app.

```bash
# default isolated file
backend/.venv/bin/python backend/eval/run_locomo_memory_eval.py --fresh-eval-db ...

# explicit path
EVAL_DB_PATH=/tmp/locomo_eval.db backend/.venv/bin/python backend/eval/run_locomo_memory_eval.py ...

# only if you intentionally want the production DB
backend/.venv/bin/python backend/eval/run_locomo_memory_eval.py --use-main-db ...
```
