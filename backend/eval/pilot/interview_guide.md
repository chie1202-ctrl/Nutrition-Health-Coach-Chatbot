# NutriCoachAI — Semi-Structured Interview Guide

**When:** End of Session 2, after questionnaires (~15 minutes)  
**Format:** Semi-structured; follow probes where answers are brief  
**Recording:** Audio only with explicit consent; store outside git  
**Status:** Ready for use; **sessions not yet conducted** (pending recruitment)

---

## Opening (1 min)

> Thank you for trying NutriCoachAI twice. There are no right or wrong answers. I'm interested in your honest experience using it as an everyday wellness coach, not as medical treatment. May I record this conversation for my notes? [If no: take written notes only.]

---

## Block A — Cross-session memory (4 min)

1. **Recall task:** When you asked about your goal and foods to avoid in the second session, did the coach's answer match what you remembered telling it the first time?
   - *Probe:* Did anything important seem missing or wrong?
2. Did you look at the **Memory Viewer**? If yes, did what you saw there match what the coach said in chat?
   - *Probe:* Would showing summaries change how much you trust the coach?
3. Compared to starting fresh each time, did the second session feel like continuing a conversation?

**Coding themes:** `memory_accurate` | `memory_gap` | `memory_viewer_helpful` | `memory_viewer_unused`

---

## Block B — Food-choice and everyday use (4 min)

4. Tell me about the **pizza vs stir-fry** comparison. Was the side-by-side advice useful for a real takeaway decision?
   - *Probe:* What would make comparisons more useful (portion size, specific restaurant, cost)?
5. You mentioned eating out in Session 1. Did the coach use that context when you asked follow-up questions?
6. How does this compare to searching nutrition advice online or using a generic chatbot?

**Coding themes:** `food_choice_helpful` | `food_choice_too_generic` | `eating_out_recalled` | `prefer_other_tools`

---

## Block C — Trust, privacy, and boundaries (4 min)

7. The app says data stays local. Did that matter to you when entering weight and chat content?
   - *Probe:* Any concerns about running Ollama locally (setup, speed)?
8. Did you notice **source citations** under any answers? Did they increase confidence in the advice?
9. Where is the line between helpful coaching and something you would want a clinician for?
   - *Probe:* Did the medical disclaimer feel sufficient?

**Coding themes:** `privacy_valued` | `privacy_skeptical` | `rag_trust` | `boundary_aware` | `disclaimer_adequate`

---

## Block D — Usability and willingness to reuse (2 min)

10. What was the hardest part of using the system (setup, waiting for replies, navigation)?
11. On a scale of 1–10, how likely would you be to use this weekly for lifestyle nutrition — not for diagnosing illness?
12. If you could change one thing before recommending it to a friend, what would it be?

**Coding themes:** `latency` | `setup_friction` | `ui_confusion` | `reuse_positive` | `reuse_negative`

---

## Closing (1 min)

> Is there anything else about the coach, memory, or food comparisons you want me to know?

Thank participant; remind them data is pseudonymised and they may withdraw consent for interview use in writing.

---

## Facilitator post-interview checklist

- [ ] Tag 3–5 salient quotes (paraphrase if needed for anonymity)
- [ ] Map quotes to themes above
- [ ] Note any safety guardrail or disclaimer incidents
- [ ] Transfer summary to participant JSON (`interview.themes`, `interview.exemplar_quotes`) in results folder
