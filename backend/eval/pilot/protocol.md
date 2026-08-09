# NutriCoachAI — User Pilot Study Protocol

**Version:** 0.1 (draft)  
**Status:** Materials prepared; **execution pending participant recruitment** (as of 2026-06-29).  
**Location:** `backend/eval/pilot/`  
**Related thesis section:** Chapter 5 — User Pilot Study

---

## 1. Purpose

Complement the controlled M0–M3 memory ablation harness (`backend/eval/run_memory_eval.py`) with **ecological validity** evidence from real users completing scripted coaching tasks over two sessions. This pilot addresses SQ3: whether participants can achieve cross-session coaching coherence and everyday food-choice decisions in natural interaction.

This is an **exploratory** low-risk wellness-technology study (n = 3–5), not a clinical trial.

---

## 2. Design summary

| Parameter | Value |
|-----------|-------|
| Design | Within-subjects, two-session longitudinal pilot |
| Target *n* | 3–5 participants |
| Recruitment | Self-recruited convenience sample (peers, family, health-interested volunteers) |
| Exclusion | Clinical patients under active medical nutrition therapy; minors without guardian consent |
| Sessions | 2 per participant, 25–35 minutes each |
| Inter-session interval | 3–7 calendar days |
| Environment | Participant's machine or supervised lab machine running `./start.sh` |
| Production config | `deepseek-r1:8b`, `OLLAMA_REASONING=false`, `MEMORY_MODE=M2`, RAG on |
| Facilitator | Researcher (optional presence for onboarding only; tasks are self-guided) |

---

## 3. Ethical considerations

- **Informed consent** before Session 1 (verbal + signed one-page form; template in supervisor appendix).
- **Right to withdraw** at any time without penalty; partial data may be retained if consented.
- **Data handling:** Assign pseudonymous IDs (`P01`–`P05`); no real names in `results.json`.
- **Audio recording** for post-session interview only, with explicit opt-in.
- **Non-medical positioning:** Participants read Settings → About disclaimer; facilitator reminds that the system is a research prototype, not medical advice.
- **Ethics approval:** Confirm with supervisor whether formal faculty ethics exemption is required for low-risk self-recruited pilot (MSc common practice).

---

## 4. Session flow

### Session 1 (~30 min)

| Step | Task | Success criterion | Notes |
|------|------|-------------------|-------|
| S1-T1 | Onboarding wizard | Profile, goal, diet, allergies saved | Use fresh pseudonymous account |
| S1-T2 | Goal & habit chat | ≥3 user turns on weight goal and eating habits | Include statement: *"I often eat out / takeaway several times a week"* |
| S1-T3 | Weight log | One weight entry recorded | Verify dashboard BMI/REE update |
| S1-T4 | Memory Viewer | Open viewer; locate session summary | Observer notes whether participant finds summaries |
| S1-T5 | Optional meal plan | Generate 7-day plan if time permits | Not required for pass/fail |

**Close Session 1:** Use "New Conversation" or idle timeout so session summary is generated before Session 2.

### Session 2 (~30 min)

| Step | Task | Success criterion | Notes |
|------|------|-------------------|-------|
| S2-T1 | Cross-session recall | Ask: *"What was my weight goal and what foods should I avoid?"* | Coach reply mentions goal magnitude and stated allergy/diet constraint |
| S2-T2 | Preference recall | Ask: *"What did we discuss about my eating-out habits last time?"* | Reply references takeaway/dining-out context from S1 |
| S2-T3 | Food-choice comparison | Ask: *"I'm choosing between pizza and a Chinese vegetable stir-fry for dinner — which fits my goal better?"* | Structured comparison card or equivalent A vs B guidance |
| S2-T4 | RAG transparency | Ask one nutrition guideline question | ≥1 source chip visible under reply |
| S2-T5 | Questionnaires | SUS + custom trust/usefulness (see `questionnaire.json`) | Immediately after tasks |
| S2-T6 | Semi-structured interview | 15 min (see `interview_guide.md`) | Audio if consented |

---

## 5. Task scoring (facilitator rubric)

Record per task: `pass` | `partial` | `fail` | `not_attempted`.

| Task ID | Pass | Partial | Fail |
|---------|------|---------|------|
| S2-T1 | Goal + restriction both correct | One of two | Neither or hallucinated |
| S2-T2 | Recalls eating-out context | Vague generic advice | No memory of prior session |
| S2-T3 | Compares both options with goal-linked rationale | Single-option advice only | Refusal, unsafe, or ignores profile |
| S2-T4 | ≥1 RAG source chip | Answer grounded but no chip | No grounding / wrong topic |

**Overall task success rate** = passes / attempted scored tasks.

---

## 6. Materials checklist

- [ ] `./start.sh` verified; Ollama + RAG ready (`GET /health`)
- [ ] Consent form (paper or PDF)
- [ ] Printed or on-screen task script (this file §4)
- [ ] `questionnaire.json` (Google Form / paper / local HTML — facilitator choice)
- [ ] `interview_guide.md`
- [ ] Empty copy of `results_template.json` per participant
- [ ] Screen recording optional (with consent) for thesis exemplar quotes

---

## 7. Data collection and storage

1. After each participant completes Session 2, fill `results/<pilot_run_YYYYMMDD>/P0X.json` from `results_template.json`.
2. Store interview notes in `results/<pilot_run_YYYYMMDD>/P0X_interview.md` (no audio filenames in git).
3. Aggregate summary: `results/<pilot_run_YYYYMMDD>/summary.json` (mean SUS, task pass rates).
4. **Do not commit** raw audio, signed consent scans, or real names.

---

## 8. Execution status

| Milestone | Status |
|-----------|--------|
| Protocol drafted | ✅ 2026-06-29 |
| Questionnaire prepared | ✅ |
| Interview guide prepared | ✅ |
| Results template prepared | ✅ |
| Participant recruitment | ⬜ Pending |
| Sessions conducted | ⬜ Pending |
| Thesis Ch5 results populated | ⬜ Pending data |

---

## 9. Relationship to controlled evaluation

| Aspect | M1/M1.5 harness | This pilot |
|--------|-----------------|------------|
| *n* | 3 scripts × 4 modes | 3–5 humans × 2 sessions |
| RAG | Off (isolation) | On (production path) |
| Metrics | Keyword pass rate, fidelity, tokens | Task rubric, SUS, qualitative themes |
| Claim strength | Reproducible ablation | Exploratory ecological evidence |

See Chapter 6 for limitations and how pilot qualitative findings complement (but do not replace) harness metrics.
