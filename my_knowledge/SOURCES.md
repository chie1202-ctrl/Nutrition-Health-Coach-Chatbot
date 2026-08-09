# RAG knowledge base — source catalogue (C11)

Curated PDFs for Chroma indexing via `initialize_rag()` in `backend/logic.py`.  
**Re-index after adding or removing files:** delete `backend/database/vector_db/` and restart `./start.sh`.

| File | Title | Publisher | URL | Licence / use |
|------|-------|-----------|-----|----------------|
| `WHO_Healthy_Diet_Fact_Sheet_394.pdf` | Healthy diet (Fact sheet N°394) | World Health Organization | https://www.who.int/news-room/fact-sheets/detail/healthy-diet | WHO fact sheet; cite WHO; non-commercial research / education |
| `CDC_Sodium_and_Dietary_Guidelines.pdf` | Get the Facts: Sodium and the Dietary Guidelines | U.S. CDC | https://stacks.cdc.gov/view/cdc/106415 | U.S. government public domain |
| `CDC_Eat_Well_Away_from_Home.pdf` | Eat Well Away from Home (DPP participant module) | U.S. CDC | https://www.cdc.gov/diabetes-prevention/media/pdfs/legacy/Participant-Module-15_Eat_Well_Away_from_Home.pdf | U.S. government public domain |
| `NHS_Eatwell_Guide_Booklet_2018.pdf` | The Eatwell Guide booklet | UK Office for Health Improvement and Disparities (PHE) | https://www.gov.uk/government/publications/the-eatwell-guide | Open Government Licence v3.0 |
| `Dietary Guidelines for Americans, 2020–2025.pdf` | Dietary Guidelines for Americans, 2020–2025 | USDA / HHS | https://www.dietaryguidelines.gov | U.S. government public domain |
| `2025-2030美國膳食指南.pdf` | Dietary Guidelines for Americans (Chinese edition) | USDA / HHS | https://www.dietaryguidelines.gov | U.S. government public domain |
| `Nutritive Value of Foods.pdf` | Nutritive Value of Foods | USDA Agricultural Research Service | https://www.ars.usda.gov | U.S. government public domain |
| `physical activity and sedentary behaviour.pdf` | Physical activity and sedentary behaviour guidelines | WHO | https://www.who.int | WHO publication |
| `motivational-interviewing-guide.pdf` | Motivational interviewing guide | Various (training material) | — | Verify local use; coaching communication reference |

**Added (C11 expansion):** 2026-07-01 — WHO, CDC sodium, CDC away-from-home, NHS Eatwell (4 files).  
**Corpus size:** 9 PDFs.

**RAG role:** supplementary evidence for chat, food-choice comparison, and meal-plan prompts. User profile allergies and medical conditions remain authoritative in `logic.py` prompt rules.
