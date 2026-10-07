# Dual- versus Single-Suggestion AI Support for Radiographic Interpretation in Residents: Randomized Multireader Study

Code and data for the 60-case set used in a prospective, multicenter, randomized three-arm reader study of dual- versus single-suggestion AI support.

## Abstract

**Purpose:** To compare dual- and single-suggestion AI support for radiographic interpretation by residents, particularly when the shared AI suggestion was incorrect.

**Materials and Methods:** This prospective, multicenter, randomized three-arm reader study was conducted at three hospitals in China from July to September 2026 (ChiCTR2600129243). After specialty stratification, 132 residents with fewer than 3 years of clinical experience were randomized 1:1:1 to GPT-5.4 alone (group A), GPT-5.4 plus Kimi-K2.6 (group B), or GPT-5.4 plus Gemini-3.6 Flash (group C); 123 were analyzed. Participants interpreted 60 radiographs before and after AI support. The primary outcome was accuracy change. Welch ANOVA and Holm-adjusted $t$ tests compared support conditions; HC3 linear models assessed specialty interaction.

**Results:** Among 123 residents (mean age, 24.1 years $\pm$ 1.4; 65 women), radiology residents showed greater accuracy improvement with dual- than single-suggestion support (B–A, 6.69 percentage points [95\% CI, 0.97–12.40]; C–A, 7.87 percentage points [95\% CI, 1.64–14.11]; Holm-adjusted $P$ = .030 for both), whereas accuracy change did not differ in non-radiology residents ($P$ = .20). When GPT-5.4 was incorrect, AI-assisted accuracy was higher with dual- than single-suggestion support in radiology residents (40.1\% and 40.4\% vs 20.0\%) and non-radiology residents (31.3\% and 31.0\% vs 12.1\%) (all Holm-adjusted $P$ < .001). The dual-suggestion effect differed by specialty (interaction difference, 10.44 percentage points; 95\% CI, 4.36–16.52; $P$ < .001).

**Conclusion:** Dual-suggestion support may mitigate the influence of erroneous AI suggestions, with greater accuracy improvement observed in radiology but not non-radiology residents.

## Repository contents

This repository reproduces the AI-suggestion arm of the study: three independent multimodal models generated a primary diagnosis and supporting rationale from the same deidentified radiograph and clinical history. Model outputs were generated once, locked before case selection, and used unchanged in the reader study. Each model was correct on 39 of 60 cases (65.0%).

| Path | Description |
| --- | --- |
| `data/cases.csv` | Case IDs, body region, clinical history, and reference diagnoses |
| `data/lexicon.csv` | Prespecified diagnostic lexicon used for free-text scoring |
| `data/images/` | Case radiographs (`Case_01.jpg`–`Case_60.jpg`; not required to score locked outputs) |
| `run_api.py` | Retest the three models with the recorded study settings |

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
```

Recorded settings: GPT-5.4 and Kimi-K2.6 via Azure OpenAI Chat Completions (temperature 0.2 / 4096 tokens / JSON object, and 0.6 / 4096 tokens / prompt-enforced JSON, respectively); Gemini-3.6 Flash via Vertex `generateContent` (temperature 0.2 / 800 tokens / `application/json`). Prompt keys are `primary_diagnosis`, `secondary_diagnosis`, and `basis`. Three attempts per case.
