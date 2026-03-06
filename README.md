# ProbID Decision Assistant API (FastAPI)

This backend scaffold was generated from the files you uploaded in `ProbID/`:

- `ProbID/ProbIDTool.tsx`
- `ProbID/FaganChart.tsx`
- `ProbID/references/methods/page.tsx`
- `ProbID/lrMath.ts`
- `ProbID/probidDecision.ts`
- `ProbID/lrSyndromes.ts`

It implements the same core decision pattern shown in your UI:

- Bayes updates using likelihood ratios (LRs)
- pretest -> post-test probability
- decision thresholds (`observe` / `test` / `treat`)
- treatment threshold formula from your methods page:
  - `P(treat) = H(unnecessary treatment) / (H(unnecessary treatment) + H(missed diagnosis))`
- observation threshold formula from your methods page:
  - `P(observe) = 0.5 * P(treat)`

## Current status

The backend now loads your real module catalog from:

- `backend/app/data/probid_modules.json`

That JSON is generated from `ProbID/lrSyndromes.ts` using:

- `backend/scripts/extract_probid_modules.js`

The API math and harm-threshold logic are aligned to your uploaded:

- `ProbID/lrMath.ts`
- `ProbID/probidDecision.ts`

## Remaining gaps (for full parity with the UI)

- Endocarditis score auto-management logic in `ProbID/ProbIDTool.tsx` (VIRSTA / DENOVA / HANDOC)
- VAP/Endocarditis pretest risk-modifier toggles in `ProbID/ProbIDTool.tsx`
- Any future source-enrichment logic from `lrSyndromes.ts` helper wrappers (`*_WITH_SOURCES`) if you want exact source metadata parity

## Project structure

- `backend/app/main.py` - FastAPI app + endpoints
- `backend/app/schemas.py` - request/response + ProbID module schemas
- `backend/app/engine.py` - LR/Bayes + thresholds + recommendation engine
- `backend/app/services/module_store.py` - in-memory module registry
- `backend/app/data/demo_probid_modules.py` - placeholder modules

## Run locally

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open docs at `http://127.0.0.1:8000/docs`.
Open the guided web UI at `http://127.0.0.1:8000/assistant`.

## Evaluate parser quality

Run the local evaluator against the holdout cases:

```bash
cd backend
.venv311/bin/python scripts/evaluate_local_parser.py --parsers local rule --show-failures
```

You can also point it at another labeled dataset:

```bash
cd backend
.venv311/bin/python scripts/evaluate_local_parser.py --dataset app/data/parser_eval_cases.json --parsers local
```

## Smoke test newly added syndromes

Run the regression smoke test for the recently added syndrome modules and guided assistant paths:

```bash
cd backend
.venv311/bin/python scripts/smoke_test_new_syndromes.py
```

## Enable LLM extraction (optional)

`POST /v1/analyze-text` now supports a local example-driven parser in addition to the rule-based parser and the optional OpenAI-backed parser.

The local parser reads labeled examples from:

- `backend/app/data/parser_training_examples.json`

For a separate holdout set you can evaluate against:

- `backend/app/data/parser_eval_cases.json`

It uses nearest training examples to infer `moduleId`, `presetId`, and missing findings, then merges that with the rule-based extractor. The JSON file is read on each request, so you can iterate on examples without changing code.

Set environment variables before starting the API:

```bash
export OPENAI_API_KEY="your_key_here"
export OPENAI_MODEL="gpt-4.1-mini"   # optional
# export OPENAI_BASE_URL="..."        # optional (for compatible gateways)
```

Then use `parserStrategy` in the request:

- `"auto"` (default): try local parser, then OpenAI parser, then rule parser
- `"local"`: require the local example-driven parser (returns error if it fails unless `allowFallback=true`)
- `"openai"`: require OpenAI parser (returns error if it fails unless `allowFallback=true`)
- `"rule"`: use local rule-based parser only

You can also override the parser model per request with:
- `parserModel` (for example: `"gpt-4.1-mini"` or `"gpt-4.1"`)

## Endpoints

- `GET /health`
- `GET /v1/modules`
- `GET /v1/modules/{module_id}`
- `POST /v1/modules/register`
- `POST /v1/analyze`
- `POST /v1/analyze-text` (free-text parser + optional analysis)
- `POST /v1/assistant/turn` (guided "Uncertainty Assistant" conversation)

## Example analyze request

```json
{
  "moduleId": "cap",
  "presetId": "ed_adult",
  "findings": {
    "cap_fever": "present",
    "cap_cxr_consolidation": "present",
    "cap_procal_high": "absent"
  },
  "orderedFindingIds": ["cap_fever", "cap_cxr_consolidation", "cap_procal_high"],
  "includeExplanation": true
}
```

## Example with `probidControls` (UI parity features)

```json
{
  "moduleId": "endo",
  "presetId": "sab_bacteremia",
  "findings": {
    "endo_tte": "present",
    "endo_bcx_major_typical": "present"
  },
  "probidControls": {
    "endoRiskModifiers": {
      "enabled": true,
      "selectedIds": ["prosthetic_valve", "cied", "hemodialysis"]
    },
    "endoScores": {
      "virsta": {
        "enabled": true,
        "intracardiacDevice": true,
        "persistentBacteremia48h": true,
        "acquisition": "community_or_nhca"
      },
      "denova": {
        "enabled": true,
        "numPositive2": true,
        "originUnknown": true,
        "valveDisease": true
      },
      "handoc": {
        "enabled": false
      }
    }
  }
}
```

Notes:
- VAP/Endo risk modifiers are applied as an odds-space pretest multiplier with the same shrink/cap behavior used in `ProbIDTool.tsx`.
- VIRSTA suppresses `sab_context` Endo risk modifiers for pretest adjustment (matching the UI).
- Endo score auto-compute injects the `endo_*_high` LR items server-side (matching the UI).

## Example free-text request (`POST /v1/analyze-text`)

```json
{
  "text": "ED patient with fever, tachypnea, and CXR infiltrate. Procalcitonin low.",
  "parserStrategy": "auto",
  "parserModel": "gpt-4.1-mini",
  "runAnalyze": true
}
```

Example response shape (trimmed):
- `parsedRequest` -> the structured `AnalyzeRequest` the parser built
- `understood` -> human-readable summary ("what I understood")
- `warnings` -> ambiguity/low-confidence notes
- `parserFallbackUsed` -> `true` if the requested parser failed and rule parser was used
- `analysis` -> the normal `/v1/analyze` response (if `runAnalyze=true`)

## Example forcing the OpenAI parser

```json
{
  "text": "ED patient with fever, tachypnea, and CXR infiltrate. Procalcitonin low.",
  "parserStrategy": "openai",
  "allowFallback": false,
  "runAnalyze": true
}
```

The `/assistant` web UI also includes a **Parser model** field that sends this value on each turn, so you can switch models live without restarting FastAPI.

## Tips for better free-text parsing (current rule-based MVP)

- Mention the syndrome explicitly if possible (`CAP`, `endocarditis`, `VAP`, etc.)
- Mention the setting (`ED`, `ICU`, `primary care`) to improve pretest preset selection
- Use short clinical phrases:
  - `fever`, `tachypnea`, `CXR infiltrate`
  - `no fever`, `afebrile`, `clear CXR`
- If parsing is off, use `moduleHint` and/or `presetHint`

## Guided UX endpoint: `POST /v1/assistant/turn`

This endpoint provides a conversational flow that:
1. Introduces itself as **Uncertainty Assistant**
2. Asks which syndrome to approach
3. Asks for the setting/pretest context
4. Accepts a plain-language case description
5. Returns the parsed text analysis + ProbID recommendation

It is stateless on the server. The client sends back the `state` returned by the previous response.

### Turn 1 (start conversation)

```json
{}
```

### Turn 2 (choose syndrome)

```json
{
  "state": {
    "stage": "select_module"
  },
  "selection": "cap"
}
```

### Turn 3 (choose preset/context)

```json
{
  "state": {
    "stage": "select_preset",
    "moduleId": "cap"
  },
  "selection": "ed_adult"
}
```

### Turn 4 (describe case in plain language)

```json
{
  "state": {
    "stage": "describe_case",
    "moduleId": "cap",
    "presetId": "ed_adult",
    "parserStrategy": "auto",
    "allowFallback": true
  },
  "message": "ED patient with fever, tachypnea, and CXR infiltrate. Procalcitonin low."
}
```

Response includes:
- `assistantMessage` (natural prompt/summary)
- `state` (send this back next turn)
- `options` (for button-based UI)
- `analysis` (same structure as `/v1/analyze-text` on the final step)

## Refresh module JSON from `ProbID/lrSyndromes.ts`

```bash
node backend/scripts/extract_probid_modules.js
```

## Register/override module data at runtime

If you can export your frontend module data to JSON, post it to `POST /v1/modules/register` using this shape:

```json
{
  "modules": [
    {
      "id": "cap",
      "name": "Community-Acquired Pneumonia",
      "pretestPresets": [{ "id": "ed", "label": "ED", "p": 0.2 }],
      "defaultHarms": {
        "unnecessary_treatment": 1,
        "missed_diagnosis": 5
      },
      "items": [
        {
          "id": "cxr_infiltrate",
          "label": "CXR infiltrate",
          "category": "imaging",
          "lrPos": 8.5,
          "lrNeg": 0.2
        }
      ]
    }
  ]
}
```
