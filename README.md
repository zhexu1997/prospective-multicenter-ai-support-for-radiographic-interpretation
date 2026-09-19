# Dual- versus Single-Suggestion AI Support for Radiographic Interpretation in Less Experienced Physicians

Code and locked AI outputs for the 60-case set used in a prospective, multicenter, randomized three-arm reader study of dual- versus single-suggestion AI support.

## Abstract

**Background:** Less experienced physicians may be susceptible to erroneous AI suggestions during radiographic interpretation. Whether dual-suggestion support can mitigate the influence of erroneous AI suggestions remains unclear.

**Purpose:** To compare dual- and single-suggestion AI support for radiographic interpretation by less experienced physicians, particularly when the shared AI suggestion was incorrect.

**Materials and Methods:** In this prospective, multicenter, randomized three-arm reader study, 123 residents with fewer than 3 years of clinical experience were assigned to GPT-5.4 alone (group A), GPT-5.4 plus Kimi-K2.6 (group B), or GPT-5.4 plus Gemini-3.6 Flash (group C) after stratification by specialty. Participants interpreted 60 chest and abdominal radiographs before and after AI assistance. The primary outcome was change in diagnostic accuracy; accuracy was also assessed by shared AI suggestion correctness. Additional outcomes included interpretation time, confidence change, and diagnostic revisions.

**Results:** Among 123 residents (mean age, 24 years ± 1; 65 women), diagnostic accuracy increased after AI assistance in all three support conditions (all *P* < .001). For radiology residents, accuracy change was greater in groups B and C than in group A (mean differences, 6.69 and 7.87 percentage points; 95% CIs, 0.97–12.40 and 1.64–14.11, respectively; Holm-adjusted *P* = .030 for both comparisons), whereas accuracy change did not differ across support conditions in non-radiology residents (*P* = .20). When GPT-5.4 was incorrect, AI-assisted accuracy was higher with dual- than single-suggestion support in both resident groups (both Holm-adjusted *P* < .001). In non-radiology residents, AI-assisted accuracy was lower with dual-suggestion support when GPT-5.4 was correct (both Holm-adjusted *P* < .001). Specialty interactions were observed for accuracy change (*P* for interaction < .001) and incorrect-to-correct revisions (*P* for interaction = .003).

**Conclusion:** Dual-suggestion support may mitigate the influence of erroneous AI suggestions during radiographic interpretation by less experienced physicians. Greater improvement in diagnostic accuracy was observed among radiology residents, but not among non-radiology residents.

## Repository contents

This repository reproduces the AI-suggestion arm of the study: three independent multimodal models generated a primary diagnosis and supporting rationale from the same deidentified radiograph and clinical history. Model outputs were generated once, locked before case selection, and used unchanged in the reader study. Each model was correct on 39 of 60 cases (65.0%).

| Path | Description |
| --- | --- |
| `data/cases.csv` | Case IDs, body region, clinical history, and reference diagnoses |
| `data/ai_suggestions.csv` | Locked GPT-5.4, Kimi-K2.6, and Gemini-3.6 Flash outputs |
| `data/lexicon.csv` | Prespecified diagnostic lexicon used for free-text scoring |
| `data/images/` | Case radiographs (`Case_01.jpg`–`Case_60.jpg`; not required to score locked outputs) |
| `matching.py` | Lexicon matching for primary-diagnosis correctness |
| `evaluate.py` | Score predictions against reference diagnoses |
| `run_api.py` | Retest the three models with the recorded study settings |
| `results/` | Accuracy summaries written by `evaluate.py` |

Group A received GPT-5.4 alone. Group B received GPT-5.4 plus Kimi-K2.6. Group C received GPT-5.4 plus Gemini-3.6 Flash. Participant-level reader data are not included here.

## Score locked outputs

```bash
pip install -r requirements.txt
python evaluate.py
```

## Retest via API

Single case, same interface as the appendix GPT script:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...          # Azure OpenAI endpoint
export OPENAI_MODEL=gpt-5.4         # Azure deployment name

python run_api.py \
  --model GPT-5.4 \
  --image data/images/Case_01.jpg \
  --history "Cough and chest tightness" \
  --examination "Chest radiograph - frontal view" \
  --output gpt_case_result.json
```

All 60 cases:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export OPENAI_MODEL=gpt-5.4
export KIMI_MODEL=kimi-k2.6         # Azure deployment name; optional KIMI_API_KEY / KIMI_BASE_URL
export VERTEX_PROJECT=...
export VERTEX_LOCATION=us-central1
export VERTEX_ACCESS_TOKEN="$(gcloud auth print-access-token)"

python run_api.py
python evaluate.py --predictions results/api_suggestions.csv
```

Recorded settings: GPT-5.4 and Kimi-K2.6 via Azure OpenAI Chat Completions (temperature 0.2 / 4096 tokens / JSON object, and 0.6 / 4096 tokens / prompt-enforced JSON, respectively); Gemini-3.6 Flash via Vertex `generateContent` (temperature 0.2 / 800 tokens / `application/json`). Prompt keys are `primary_diagnosis`, `secondary_diagnosis`, and `basis`. Three attempts per case.
