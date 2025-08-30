# notebar-persona-dataset
**Persona‑Conditioned Notes for Multi‑Label Classification**

A lightweight, reproducible dataset for evaluating persona‑conditioned, multi‑label note classification. We design and release **3,173 synthetic notes** with **8,494 concept annotations**, authored under **17 personas covering all 16 MBTI types** (one type has two variants). Each note includes rich metadata (context, entities, cognitive flags) plus a *concept map* with one‑to‑many labels per note, enabling strong multi‑label benchmarks and ablations by persona.

> TL;DR: Use `summary`/`note_text` as input, predict the set of `concept_map[].concept` labels. Filter/slice by persona or MBTI to study conditioning effects.

---

## At a glance
- **Tasks:** Multi‑label note classification, persona‑conditioned evaluation, concept extraction
- **Size:** 3,173 notes, 8,494 concept annotations
- **Personas:** 17 total, spanning all 16 MBTI types
- **Format:** JSON files (one persona file contains many notes); you may flatten to JSONL
- **Splits:** Recommended (stratified by persona & label): `train/val/test = 70/10/20` (see recipe)
- **License:** CC BY 4.0 (dataset); code snippets CC0
- **PII:** Fully synthetic; no real personal data

---

## Why this exists
Persona signals (style, priorities, domain tilt) can change label distributions and model calibration. This dataset lets you:
- **Benchmark** multi‑label note classification with controlled persona variation.
- **Stress‑test** robustness across MBTI types and writer intents.
- **Ablate** by persona, MBTI, domain, and cognitive flags without privacy risk.

---

## Repository layout
Your repo may look like this (file names may vary slightly):
```
notebar-persona-dataset/
├─ data/
│  └─ notes/
│     ├─ qa_output_011_ENTJ_Naomi_Donovan.json
│     ├─ qa_output_012_ENTP_Devon_Lin.json
│     ├─ qa_output_013_ENFJ_Marcus_Ellison.json
│     ├─ qa_output_014_ESTJ_Gerald_Thompson.json
│     ├─ qa_output_015_ISTP_Riley_Reyes.json
│     ├─ qa_output_016_ISFP_Sierra_Brooks.json
│     ├─ qa_output_017_ESFP_Jordan_Parker.json
│     └─ qa_output_018_INTJ_Marcus_Rhodes.json
├─ README.md   ← you are here
└─ (optional) scripts/, schema/, docs/
```
**Naming convention:** `qa_output_<id>_<MBTI>_<First_Last>.json`

---

## Record schema (per note)
Each persona file contains an array of note objects with the fields below (fields marked *optional* may be absent).

```jsonc
{
  "schema_version": "2025-05-14",
  "note_id": 1,
  "context": { "date": "YYYY-MM-DD", "time": "HH:MM:SS", "location": "string" },
  "summary": "Short synopsis of the note.",
  "note_flags": {
    "personal": true|false,
    "professional": true|false,
    "academic": true|false,
    "other": true|false,
    "is_fragment": true|false,
    "is_imperative": true|false,
    "is_brainstorm": true|false,
    "is_journal": true|false
  },
  "cognitive_state": "focused|motivated|reflective|...",
  "thread": { "continues_previous": true|false, "shared_entities": ["string", "..."] },
  "entities": [{
    "name": "string",
    "category": "organization|location|artifact|person|...",
    "subcategory": "string",
    "life_area": "personal|professional|academic|other",
    "realm": "physical|digital|conceptual",
    "concepts": ["string", "..."]
  }],
  "concept_map": [{
    "concept": "string",                    // **label**
    "summary": "short rationale/definition",
    "kind": "task|risk|insight|theme|...",  // coarse type of concept
    "subkind": "string",
    "life_area": "personal|professional|...",
    "realm": "physical|digital|conceptual",
    "is_imperative": true|false,
    "is_journal": true|false,
    "canonical_analysis": { /* rhetorical/semantic facets */ },
    "canonical_score": { /* float scores for facets */ },
    "related_entities": ["string", "..."],
    "confidence": 0.0-1.0
  }],
  "note_text": "Full prose of the note (optional).",
  "confidence_score": 0.0-1.0
}
```
**Label space:** `concept_map[].concept` (strings). A single note can have **multiple** concepts → multi‑label classification.

---

## Quickstart
### 1) Install basics
```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas scikit-learn tqdm
```

### 2) Load & flatten to JSONL
This reads all persona JSON files and emits a flat `notes.jsonl` (one row per note) with `text` and `labels`.
```python
import json, glob
from pathlib import Path

rows = []
for fp in glob.glob("data/notes/*.json"):
    persona_file = Path(fp).name
    # Derive persona + MBTI from filename: qa_output_###_<MBTI>_<First_Last>.json
    parts = persona_file.replace(".json","").split("_", 3)
    mbti = parts[2] if len(parts) > 2 else "UNK"
    persona = parts[3] if len(parts) > 3 else "UNK"

    with open(fp, "r") as f:
        notes = json.load(f)

    for n in notes:
        text = n.get("note_text") or n.get("summary") or ""
        labels = sorted(list({ c.get("concept") for c in n.get("concept_map", []) if c.get("concept") }))
        rows.append({
            "note_id": n.get("note_id"),
            "mbti": mbti,
            "persona": persona,
            "text": text,
            "labels": labels,
            "num_labels": len(labels),
            "flags": n.get("note_flags", {}),
        })

with open("notes.jsonl", "w") as out:
    for r in rows:
        out.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Wrote {len(rows)} rows to notes.jsonl")
```

### 3) Build a simple baseline (TF‑IDF → One‑Vs‑Rest Logistic Regression)
```python
import json, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
from sklearn.model_selection import train_test_split

SEED = 2025
random.seed(SEED)

texts, labels = [], []
with open("notes.jsonl") as f:
    for line in f:
        ex = json.loads(line)
        texts.append(ex["text"])
        labels.append(ex["labels"])

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(labels)

X = TfidfVectorizer(min_df=2, ngram_range=(1,2)).fit_transform(texts)
X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)

clf = OneVsRestClassifier(LogisticRegression(max_iter=200))
clf.fit(X_tr, Y_tr)
Y_pr = clf.predict(X_te)

print("micro F1:", f1_score(Y_te, Y_pr, average="micro"))
print("macro F1:", f1_score(Y_te, Y_pr, average="macro"))
print("micro P/R:", precision_score(Y_te, Y_pr, average="micro"), recall_score(Y_te, Y_pr, average="micro"))
print("Jaccard:", jaccard_score(Y_te, Y_pr, average="samples"))
```

---

## Recommended evaluation protocol
- **Metrics:** micro/macro **F1**, micro **precision/recall**, **sample‑wise Jaccard** (IoU). Consider **subset accuracy** (exact match) for stricter comparisons.
- **Splits:** Stratify by **(MBTI, #labels)**; maintain label coverage in each split.
- **Persona conditioning:** Report per‑MBTI and per‑persona scores; optionally macro‑average across personas.
- **Ablations:** (a) without persona metadata; (b) training on subset of personas; (c) domain‑constrained slices (e.g., only `professional` notes).

---

## Personas & examples
Examples in this release include (non‑exhaustive):
- **INTJ** — Marcus Rhodes (strategy/audit‑oriented notes)
- **ENTJ** — Naomi Donovan (board/strategy/OKR notes)
- **ENTP** — Devon Lin (creative/brainstorming notes)
- **ENFJ** — Marcus Ellison (school leadership & culture memos)
- **ESTJ** — Gerald Thompson (operations/go‑live checklists)
- **ISTP** — Riley Reyes (garage/mechanical troubleshooting logs)
- **ISFP** — Sierra Brooks (nature/photography journals)
- **ESFP** — Jordan Parker (event planning & outreach)

---

## Reproducibility
- **Seed:** 2025 for random splits and baselines.
- **Determinism:** Pin scikit‑learn version in experiments; consider setting `PYTHONHASHSEED=0`.
- **Logging:** Save metrics with persona/MBTI breakdowns for apples‑to‑apples comparisons.

---

## Ethics & safety
- **Synthetic only.** No real PII or sensitive attributes are included.
- **Intended use.** Evaluation/research/teaching. Not intended for profiling real people.
- **Bias.** Personas can induce distributional shifts; report per‑slice metrics to detect performance gaps.

---

## License
- **Data:** Creative Commons **CC BY 4.0**. Cite this repo when you use the dataset.
- **Code snippets:** CC0 / public domain.

---

## How to cite
```
@dataset{notebar-persona-dataset_2025,
  title  = {notebar-persona-dataset: Persona-Conditioned Notes for Multi-Label Classification},
  author = {Note Bar},
  year   = {2025},
  url    = {[https://github.com/your-org/nb-personas](https://github.com/Open-Knowledge-of-New-York-LLC/notebar-persona-dataset/}
}
```

---

## Changelog
- **v2025.08** — Initial release: 3,173 notes, 8,494 concept annotations; 17 personas (16 MBTI types represented).

---

## Maintainers
- Maintainer: @openknowledgeny (Josh, on behalf of Note Bar) — issues and PRs welcome.
