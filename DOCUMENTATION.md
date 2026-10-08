# Industrial Defect Detector — End-to-End Documentation

Complete reference for the system: what it does, how each piece works, and how to
operate it.

**Related documents**
- [README.md](README.md) — quick setup and run commands.
- [PROJECT_REPORT.md](PROJECT_REPORT.md) — methodology, measured results,
  limitations, and decisions made during the build.

This document is the operating manual. Where it states a number, that number was
measured; where something is unverified, it says so.

---

## Table of contents

1. [What the system does](#1-what-the-system-does)
2. [Feature overview](#2-feature-overview)
3. [How it works](#3-how-it-works)
4. [Installation](#4-installation)
5. [User guide: the web interface](#5-user-guide-the-web-interface)
6. [User guide: the HTTP API](#6-user-guide-the-http-api)
7. [User guide: training and evaluation](#7-user-guide-training-and-evaluation)
8. [User guide: onboarding a new product](#8-user-guide-onboarding-a-new-product)
9. [User guide: verification tools](#9-user-guide-verification-tools)
10. [Configuration reference](#10-configuration-reference)
11. [File and data layout](#11-file-and-data-layout)
12. [Interpreting the output](#12-interpreting-the-output)
13. [Troubleshooting](#13-troubleshooting)
14. [Limits — what this system cannot do](#14-limits--what-this-system-cannot-do)

---

## 1. What the system does

You give it a photograph of a manufactured part. It tells you three things:

1. **Is this part defective?** A pass/fail verdict with a numeric anomaly score.
2. **Where is the defect?** A heatmap and an outlined region.
3. **What kind of defect is it?** A named defect type with probabilities across
   that product's known failure modes.

The detection stage learns only from **defect-free** examples, so it flags anything
unusual — including failure modes nobody labelled. A second, supervised stage
attaches a name to what was found, when labelled examples exist.

Seven product categories are trained: `bottle`, `capsule`, `carpet`, `grid`,
`metal_nut`, `tile` (from the MVTec AD dataset) and `wood` (added through the
onboarding workflow). New products can be added from a folder of photos without
writing code.

---

## 2. Feature overview

| # | Feature | What it gives you | Where |
| --- | --- | --- | --- |
| 1 | **Anomaly detection** | Pass/fail verdict per image, with a calibrated threshold | §3.2 |
| 2 | **Defect localisation** | Heatmap overlay + outlined defect region + `defect_area_pct` | §3.3 |
| 3 | **Defect-type classification** | Named defect type with a full probability vector | §3.4 |
| 4 | **Leakage-free threshold calibration** | Thresholds derived from held-out good parts, never from test labels | §3.5 |
| 5 | **Wrong-product safety check** | Returns `uncertain` instead of a confident wrong answer when the photo is not the selected product | §3.6 |
| 5b | **Multi-category serving** | 7 products behind one API, one shared backbone, hot-swapped memory banks | §3.7 |
| 6 | **New-product onboarding** | Add a product from a folder of photos, one command, no code | §8 |
| 7 | **Graceful degradation** | Categories without labelled defects serve detection only, no errors | §8.4 |
| 8 | **Web interface** | Drag-and-drop upload, sample gallery, category switcher, comparison table | §5 |
| 9 | **HTTP API** | 7 endpoints, OpenAPI docs at `/docs` | §6 |
| 10 | **Cross-category comparison** | One table across every trained category | §7.4 |
| 11 | **Calibration health checks** | Warns when a category's calibration is statistically too thin | §8.5 |
| 12 | **Verification suite** | Smoke test, API test, visual montages, domain-shift test | §9 |

---

## 3. How it works

### 3.1 The pipeline, in order

```
photo
  │
  ├─ 1. Load ─────────────── decode to RGB
  ├─ 2. Preprocess ───────── resize 256×256 (bilinear, antialias) → ImageNet normalise
  ├─ 3. Backbone ─────────── frozen wide_resnet50_2, take layer2 + layer3
  ├─ 4. Patch embedding ──── 3×3 avg-pool, upsample layer3 to layer2, concat
  │                          → 1024 patches × 1536 dims on a 32×32 grid
  ├─ 5. Nearest neighbour ── distance from each patch to the category's memory bank
  │                          → 32×32 score map + one image score
  ├─ 6. Anomaly map ──────── upsample to 256×256, Gaussian blur (σ=4)
  │
  ├─ 7. VERDICT ──────────── score > image_threshold ?
  │        │
  │        ├─ no  ──────────► PASS. Stop. No defect type.
  │        │
  │        └─ yes ──────────► DEFECT
  │                             ├─ 8. Region: anomaly_map ≥ pixel_threshold
  │                             ├─ 9. Classify: attention-pooled embedding
  │                             │      + 13 geometry features → PCA → logistic regression
  │                             └─ 10. Render: heatmap + outline
  │
  └─ 11. Response ────────── JSON: score, verdict, severity, area, type,
                             probabilities, 3 images, latency
```

Steps 3–6 run anomalib's `PatchcoreModel` directly. Steps 7–11 are this project's
code.

### 3.2 Anomaly detection (PatchCore)

PatchCore is **unsupervised**: it never sees a defect during training.

**Training** takes the defect-free photos, runs them through a frozen ImageNet
backbone, and collects every patch embedding. That collection is then reduced by
*coreset subsampling* — a greedy k-centre algorithm keeps 10% of the patches chosen
to cover the space evenly. The result is the **memory bank**: a compact summary of
what "normal" looks like for that product.

**Inference** computes each patch's distance to its nearest neighbour in the bank.
Far-from-anything patches are anomalous. The image score is PatchCore's re-weighted
maximum over patches.

Why this works for industrial inspection: you usually have hundreds of good parts
and only a handful of defects, and the defects you have are not the ones you will
see next month. Learning "normal" generalises to unseen failure modes in a way that
learning specific defect classes does not.

Settings are identical for all seven categories — `wide_resnet50_2`,
`layer2 + layer3`, 10% coreset, 9 neighbours, 256×256 input, one epoch. This was
verified as adequate rather than assumed; see PROJECT_REPORT §4 for the
object-vs-texture check.

### 3.3 Localisation

The 32×32 patch-score grid is upsampled to 256×256 and Gaussian-blurred (σ=4) into
the anomaly map. Two renderings come from it:

- **Heatmap** — the map blended over the photo, blue→red. The colour scale is
  **fixed per category** (from `heatmap_range` in `metrics.json`, saturating at
  twice the pixel threshold), so a red patch means the same severity on every
  upload, not merely "the hottest thing in this image".
- **Outline** — the region where the map crosses `pixel_threshold`, drawn as a 3 px
  red contour with a light interior wash.

`defect_area_pct` is that region's share of the frame. **It overstates** — the
upsampling and blur dilate the region. See §12.3.

### 3.4 Defect-type classification

PatchCore cannot name what it found; it has no vocabulary for defects. A second
model supplies one, trained on the labelled defect folders, and it runs **only on
images stage 1 already flagged**.

Each flagged region is described two ways:

- **Appearance (1536-d)** — the same PatchCore patch embeddings, pooled using the
  anomaly map as attention weights (raised to a power so pooling concentrates on the
  hot region rather than averaging the whole part). Reusing the detector's features
  avoids a second backbone.
- **Geometry (13-d)** — area fraction, max/mean/std score in the region, connected
  component count, largest-component fraction, compactness, bounding-box aspect and
  fill, eccentricity, centroid offset, and two whole-map percentiles.

Then: scale each block independently → PCA(40) on appearance only → multinomial
logistic regression (`C=0.2`, balanced class weights).

The model is deliberately small because the data is small — 57 to 109 labelled
defects per category. Class counts are read from the folder structure, so 3, 4 and 5
defect types all work with no code change.

**Sample-size guards.** The classifier refuses to report an accuracy it cannot
support: below 5 images in the rarest class the fold count is reduced and the result
is marked indicative; below 2, cross-validation is suppressed; below 7, nested CV is
skipped. Status strings land in `metrics.json` as `cv_status` / `nested_cv_status`
and are shown in the UI.

### 3.5 Threshold calibration

This is the part most easily got wrong, so it is worth understanding.

PatchCore **memorises its training images** — they score at near-zero distance to a
bank built from them. So training images cannot calibrate a threshold. anomalib's
default sidesteps this by calibrating on the *test* split, which is test-set
leakage and would invalidate every reported number.

Instead: **10% of the good photos are held out of the memory bank**, and both
thresholds come from those alone. The test split is scored exactly once, at the end.

Two rules are computed over those held-out good images:

| Rule | Definition | Behaviour |
| --- | --- | --- |
| `max` | highest score any held-out good image produced | The maximum of ~20 samples — whatever the single strangest good image scored. Noisy. |
| `robust` *(default)* | `median + 3σ`, σ from the median absolute deviation | Estimates the same upper tail without depending on one observation. |

Across the six built-in categories the robust rule wins on mean detection accuracy
(0.961 vs 0.927) and is never worse. **But its advantage depends on having enough
calibration images** — on `wood`, calibrated from only 19, the max rule was actually
better (3 false positives vs 5). Below ~20 images the MAD is too noisy to trust.

Both operating points are always recorded (`at_threshold`, `at_max_threshold`) so
the disagreement is visible rather than hidden.

The **pixel threshold** uses the max rule: the highest value any pixel of a held-out
good image produced. Here the max is over ~1.3M pixel values rather than 20 images,
so it is a far more stable estimator.

### 3.6 Wrong-product safety check

A third verdict, `uncertain`, catches photos that are not the selected product at
all. Without it, a clean carpet photo sent to the bottle checker returned
`DEFECT — broken_small — 100% confidence`.

**It triggers on coverage, not score.** A genuine defect is *local* — the flaw
covers part of the product and the rest still matches the memory bank. A wrong
product is unfamiliar *everywhere*. Measured across all seven categories, the score
does not separate the two (tile's worst genuine defect scores 85.3; wrong-product
photos start at 55.1), but coverage does: the worst genuine defect covers 0.739 of
the frame, the least-unfamiliar wrong product covers 0.753.

The threshold is **0.80 coverage**, held in `metrics.json` as
`out_of_distribution_threshold` and configurable via
`config.OOD_COVERAGE_THRESHOLD`. It is a deliberate constant rather than a fitted
value — every held-out good image sits at exactly 0.000 coverage, so there is no
spread to calibrate from.

When it fires, the response keeps the score and all three images but returns no
defect type, because the classifier has only ever seen defects of the selected
product.

Measured: **0 of 555** genuine defects downgraded; **98%** of wrong-product photos
caught. See PROJECT_REPORT §5.2 for the full table and the limits — in particular,
it catches *wildly* different products, not similar-looking wrong ones.

### 3.7 Multi-category serving

All categories share one frozen ImageNet backbone — the weights are identical, only
the memory bank differs. `InspectionService` therefore keeps **one torch module per
distinct PatchCore configuration** and swaps the memory bank when the requested
category changes. Seven separate inferencers would duplicate a ~270 MB backbone
seven times, which does not fit alongside the banks on a 4 GB card.

Memory banks load **lazily**, on first use of each category, and are then cached.
The first request for a new category pays a one-off load; steady state is ~300 ms.

Categories are **discovered from disk** by scanning `models/*/patchcore.pt`, so a
newly onboarded category is picked up with no code change and no registration —
only an API restart.

---

## 4. Installation

Requires Python 3.12, Node 18+, and ideally an NVIDIA GPU. Verified on Python
3.12.3, Windows 11, GTX 1650 (4 GB).

```powershell
cd C:\Users\22301506\Documents\defect-detector

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# CUDA build FIRST — plain `pip install torch` silently gives CPU-only wheels.
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 `
  --index-url https://download.pytorch.org/whl/cu124

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

cd frontend; npm install; cd ..
```

Verify CUDA is live:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

`False` is workable — everything runs on CPU, roughly 3–4× slower per inspection.

**Dataset.** `src/config.py: DATASET_ROOT` points at the MVTec folders. It is only
needed for training and for the sample gallery; serving uses the saved models.

---

## 5. User guide: the web interface

### Starting it

Two terminals, both from the project root:

```powershell
# Terminal 1 — API
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --port 8000

# Terminal 2 — UI
cd frontend; npm run dev
```

Open **http://localhost:5173**.

### The layout

**Top bar** — product name, category count, and the device in use (`CUDA`/`CPU`).
The **Compare all categories** button on the right swaps the view for the
cross-category table.

**Category bar** — every trained category, grouped into *Objects* and *Textures*,
each showing its image AUROC. Click to switch. A **purple dot** marks a category
added through onboarding rather than shipped with the project.

> Switching category clears the current result. This is deliberate: thresholds,
> defect classes and the memory bank are all category-specific, so a result from one
> category means nothing under another.

**Left panel — Inspect a part**
- Five metric chips for the selected category: image AUROC, pixel AUROC, threshold,
  defect-type accuracy, and defect class count. `—` or `n/a` means that metric was
  not measurable, not that it was zero.
- A **drop zone**: drag an image in, or click to browse.
- A **sample gallery**: two held-out test images per class, green-tagged for good
  and red for each defect type. One click inspects.

**Right panel — Results**
- A **verdict banner**: green `PASS` or red `DEFECT`, with the score and threshold
  spelled out, and the inference latency.
- A **score bar** showing where this image's score sits, with the threshold marked.
- Three **image tabs**: Heatmap, Defect outline, Original.
- Two **stats**: defect area and severity.
- For a defect: the **likely defect type** with a ranked probability bar per class.

### The comparison view

Click **Compare all categories**. Every trained category with its image AUROC, pixel
AUROC, detection accuracy, defect class count and classifier accuracy. Onboarded
categories carry an `onboarded` pill. Click any row to jump to that category.

Cells reading `no labelled defects` or `insufficient samples` mean exactly that —
the number was not computed because the data could not support it.

### What you cannot do here

Onboarding a new product is **not** in the UI. It is a developer command (§8).

---

## 6. User guide: the HTTP API

Base URL `http://127.0.0.1:8000`. Interactive OpenAPI docs at `/docs`.

CORS is pre-allowed for `localhost:5173` and `localhost:4173` only; add origins in
`src/api/main.py` if you serve the UI elsewhere.

### `GET /health`

```json
{ "status": "ok", "model_loaded": true,
  "categories": ["bottle","capsule","carpet","grid","metal_nut","tile","wood"] }
```

### `GET /categories`

Every trained category with thresholds, classes and metrics. Key fields:

| Field | Meaning |
| --- | --- |
| `type` | `object` or `texture` |
| `onboarded` | `true` if added via the onboarding script |
| `has_classifier` | `false` → this category will never return a defect type |
| `threshold` | the image-score decision threshold |
| `score_range` | `[min, max]` for rendering a score bar |
| `defect_classes` | the names this category can predict (empty if no classifier) |
| `metrics` | AUROCs, operating point, counts, classifier block |

### `GET /metadata?category=<name>`

One category's summary, same shape as above. Omit `category` for the default.
Unknown category → `404`.

### `GET /comparison`

The cross-category table as an array, mirroring `outputs/comparison.json`.
`404` if you have not run `python -m src.compare`.

### `GET /samples?category=<name>`

Up to two test images per class, with base64 thumbnails.

```json
[{ "id": "wood__color__000", "category": "wood", "label": "color",
   "thumbnail": "data:image/jpeg;base64,..." }]
```

Returns `[]` — not an error — for a category with no test split.

### `POST /predict`

Multipart upload. Fields: `file` (required), `category` (optional, defaults to
`bottle`).

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -F "file=@C:\path\to\part.png" -F "category=wood"
```

Errors: `400` empty or unreadable image, `413` over 20 MB, `404` unknown category,
`503` model still loading.

### `POST /predict/sample/{sample_id}`

Runs a gallery sample. The id encodes its category (`wood__color__000`), so no
`category` field is needed. Unknown id → `404`.

### Prediction response

```jsonc
{
  "category": "wood",
  "verdict": "defect",           // "pass" | "defect" | "uncertain"
  "message": null,               // set only for "uncertain"
  "anomaly_score": 70.96,        // raw distance; compare against threshold
  "threshold": 33.79,            // this category's decision threshold
  "is_defective": true,
  "severity": 1.0,               // 0 at threshold, 1 at twice threshold
  "defect_type": "color",        // null if no classifier or if PASS
  "defect_probabilities": {      // null if no classifier or if PASS
    "color": 0.99, "combined": 0.004, "hole": 0.003,
    "liquid": 0.002, "scratch": 0.001
  },
  "defect_area_pct": 12.49,      // % of frame above pixel threshold — overstates
  "images": {                    // base64 PNG data URIs, ready for <img src>
    "original": "data:image/png;base64,...",
    "heatmap":  "data:image/png;base64,...",
    "outline":  "data:image/png;base64,..."
  },
  "latency_ms": 402.0
}
```

**Handling a category with no classifier:** `defect_type` and
`defect_probabilities` are `null` while `is_defective` may still be `true`. Check
`has_classifier` from `/categories`, or simply null-check both fields.

**Handling `verdict`:** three values, and `is_defective` is unchanged for existing
consumers.

| `verdict` | `is_defective` | `defect_type` | Meaning |
| --- | --- | --- | --- |
| `pass` | `false` | `null` | Normal product |
| `defect` | `true` | a name, or `null` if the category has no classifier | Real defect found |
| `uncertain` | `true` | **always `null`** | Photo does not look like this product at all — see §3.7 |

An `uncertain` result still carries the score and all three images, so the operator
can see what triggered it. It never carries a defect type.

---

## 7. User guide: training and evaluation

Three stages run in order; each consumes the previous one's output.

### 7.1 Everything at once

```powershell
.\scripts\train_all.ps1                          # all built-in categories
.\scripts\train_all.ps1 -Categories tile,grid    # a subset
.\scripts\train_all.ps1 -SkipTrained             # reuse existing memory banks
```

Or via Python, which is what the script wraps:

```powershell
.\.venv\Scripts\python.exe -m src.run_all --categories tile grid --nested-cv
```

`run_all` catches a failure in one category, reports it, and continues with the
rest rather than losing the batch.

### 7.2 Stage 1 — PatchCore

```powershell
.\.venv\Scripts\python.exe -m src.train_patchcore --category tile --batch-size 8
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--category` | `bottle` | which product |
| `--batch-size` | `8` | lower it if VRAM is tight |
| `--accelerator` | `auto` | `gpu`, `cpu`, or `auto` |
| `--layers` | config | override backbone layers |
| `--backbone` | `wide_resnet50_2` | override the backbone |
| `--coreset-ratio` | `0.1` | fraction of patches kept |

Writes `models/<category>/patchcore.pt` and `outputs/<category>/splits.json`.

> **Runtime warning.** Coreset selection is the bottleneck and is very sensitive to
> VRAM headroom. On a 4 GB GTX 1650, ~180–210 training images select at ~240 it/s
> and finish in under a minute; ~250–280 images push the embedding store past the
> card and collapse to ~20 it/s — 20 to 40 minutes. This is memory thrashing, not
> compute. On a larger card the big categories run at full speed.

### 7.3 Stage 2 — Calibration

```powershell
.\.venv\Scripts\python.exe -m src.calibrate --category tile --batch-size 4
```

Scores the held-out good images, derives both thresholds, then scores the test
split once. Writes `outputs/<category>/metrics.json`.

To re-derive thresholds for every category without refitting PatchCore:

```powershell
.\.venv\Scripts\python.exe -m scripts.recalibrate_all
```

This reuses the saved memory banks, is deterministic, and preserves the existing
classifier block in `metrics.json`.

### 7.4 Stage 3 — Defect-type classifier

```powershell
.\.venv\Scripts\python.exe -m src.train_classifier --category tile --nested-cv
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--category` | `bottle` | which product |
| `--pca-components` | `40` | appearance-block PCA width |
| `--folds` | `5` | cross-validation folds |
| `--nested-cv` | off | run nested CV — **use this for any reported number** |

Without `--nested-cv` you get plain cross-validation, which is mildly optimistic
because the hyper-parameters were selected on that same CV. Nested CV re-runs the
selection inside each outer fold.

Features are cached to `outputs/<category>/classifier_features.npz`, keyed on the
pixel threshold, so repeat runs skip the forward passes. A changed pixel threshold
invalidates the cache automatically.

Writes `models/<category>/defect_classifier.joblib`.

### 7.5 Comparison table

```powershell
.\.venv\Scripts\python.exe -m src.compare
```

Prints the table and writes `outputs/comparison.json`. Object-vs-texture aggregates
deliberately exclude onboarded categories, which use a different split and may lack
masks — folding them in would quietly change what the comparison measures.

---

## 8. User guide: onboarding a new product

Adds a product the system has never seen, from a folder of photographs, with no new
code. **Developer-run, one command per client.** There is no client-facing upload
page.

### 8.1 What you need

- **50+ defect-free photos** in one folder (hard floor). **200+ recommended** —
  below that the threshold estimate is unstable and you will get false alarms.
  See §8.5.
- Optionally, **labelled defect photos**, one subfolder per defect type.

Photos should be of one product, shot on one rig, with consistent lighting and
framing — the same conditions the system will see in production.

### 8.2 Detection only (no labelled defects)

```powershell
.\.venv\Scripts\python.exe -m scripts.onboard_new_category `
    --category screws `
    --good-dir "C:\clients\acme\good_photos"
```

### 8.3 With labelled defects

```
C:\clients\acme\defects\
    bent\      img001.png img002.png ...
    rust\      img010.png ...
    scratch\   img020.png ...
```

```powershell
.\.venv\Scripts\python.exe -m scripts.onboard_new_category `
    --category screws `
    --good-dir "C:\clients\acme\good_photos" `
    --defects-dir "C:\clients\acme\defects" `
    --type object
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--category` | required | name; lower-cased, spaces → underscores |
| `--good-dir` | required | folder of defect-free photos |
| `--defects-dir` | none | folder of defect subfolders; a `good` subfolder inside is skipped |
| `--type` | `object` | `object` or `texture`; affects grouping and reporting only |
| `--batch-size` | `8` | lower if VRAM is tight |
| `--accelerator` | `auto` | `gpu`, `cpu`, `auto` |
| `--force` | off | replace an already-onboarded category of this name |

### 8.4 What the script does

1. **Stages** the photos into `data/onboarded/<category>/` in MVTec layout — 20% of
   the good photos reserved as `test/good` (unseen by the memory bank, so a false
   alarm rate is measurable), the rest as `train/good`.
2. **Writes blank ground-truth masks** for any defect images. anomalib's loader
   refuses a dataset whose defective test images have no mask, and clients do not
   supply pixel annotations. Calibration detects the masks carry no positive pixels
   and reports pixel metrics as `null` rather than a meaningless number.
3. **Runs the same three stages** the built-in categories use — no duplicated logic.
4. **Skips the classifier** entirely if no defects were supplied, and removes any
   stale one.
5. **Refreshes** `outputs/comparison.json`.

Then **restart the API** and the category appears in the switcher.

Names are checked: the six built-ins are refused. Any other name **shadows** a
same-named folder in the MVTec dataset root, and the script prints a note when that
happens.

### 8.5 Reading the calibration warnings

The script ends with warnings when the calibration is statistically thin:

```
!! CALIBRATION WARNINGS -- review before deploying this category
   - only 19 calibration images (want >= 20) ...
   - false alarm rate 10% (5/49) on good images the bank never saw - above the
     5% target. The max-rule threshold would give 3/49. Collect more good photos.
```

| Warning | Trigger | What to do |
| --- | --- | --- |
| Few calibration images | < 20 | Supply more good photos |
| Few good photos | < 200 | Supply more good photos |
| High false alarm rate | > 5% on unseen good | Supply more good photos; check the photos really are defect-free and consistently shot |

**The script does not auto-correct.** Silently widening the threshold would hide the
problem. The fix is more photographs.

### 8.6 Worked example — `wood`, measured

```powershell
.\.venv\Scripts\python.exe -m scripts.onboard_new_category --category wood --type texture `
    --good-dir "C:\Users\22301506\Documents\ComVi\wood\train\good" `
    --defects-dir "C:\Users\22301506\Documents\ComVi\wood\test"
```

| | Result |
| --- | --- |
| Input | 247 good photos; 60 defects across 5 types |
| Split | 198 staged → 179 memory bank + 19 calibration; 49 unseen good |
| Image AUROC | **0.9983** |
| Detection | accuracy 0.954, **recall 1.000** (0 missed defects), 5 false positives of 49 |
| Pixel metrics | `null` — no client masks |
| Classifier | **86.1% ± 5.8%** nested CV, 5 classes |
| Runtime | **2.1 min** |
| Warnings | 19 calibration images (<20); 10% false alarm rate (>5%) |

### 8.7 Removing an onboarded category

Delete three directories and restart the API:

```powershell
Remove-Item -Recurse -Force models\screws, outputs\screws, data\onboarded\screws
```

---

## 9. User guide: verification tools

### Smoke test — does the pipeline work end to end?

```powershell
.\.venv\Scripts\python.exe -m scripts.smoke_test
.\.venv\Scripts\python.exe -m scripts.smoke_test --categories wood --per-label 2
```

One good and one defective image per class, per category, verifying the verdict.
It **reports** misses but only **fails** below a 70% verdict hit rate — a missed
subtle defect at these deliberately conservative thresholds is a model
characteristic, not a broken pipeline.

### API test — are all the routes correct?

```powershell
.\.venv\Scripts\python.exe -m scripts.api_test     # server must be running
```

Exercises every route plus error paths (`400`, `404`), and asserts the
classifier contract both ways: a category without a classifier must never return a
defect type, and one with a classifier must.

### Visual montages — is localisation landing on the real defect?

```powershell
.\.venv\Scripts\python.exe -m scripts.render_samples
.\.venv\Scripts\python.exe -m scripts.render_samples --categories wood carpet
```

Writes `outputs/<category>/sample_results.png` — original / heatmap / outline for
one image per class. The fastest way to sanity-check a newly onboarded product.

### Domain-shift test — what happens on unexpected input?

```powershell
.\.venv\Scripts\python.exe -m scripts.domain_shift_test
```

Feeds each detector the other categories' defect-free images plus synthetic grey,
noise and gradient images. **Every detector flags 100% of them as defective.** See
§14.1 — this is the single most important behaviour to understand before deploying.

---

## 10. Configuration reference

All in `src/config.py`.

### Paths

| Setting | Default | Purpose |
| --- | --- | --- |
| `DATASET_ROOT` | `C:\Users\22301506\Documents\ComVi` | MVTec categories |
| `ONBOARDED_ROOT` | `data/onboarded` | staged client photos |
| `DATASET_ROOTS` | `[ONBOARDED_ROOT, DATASET_ROOT]` | search order — **onboarded wins** |
| `MODELS_DIR` | `models/` | memory banks and classifiers |
| `OUTPUTS_DIR` | `outputs/` | metrics, splits, montages |

### Image pipeline

| Setting | Default |
| --- | --- |
| `IMAGE_SIZE` | `(256, 256)` |
| `IMAGENET_MEAN` / `IMAGENET_STD` | standard ImageNet statistics |

Changing `IMAGE_SIZE` requires retraining every category — the memory bank is tied
to the input resolution.

### PatchCore

| Setting | Default | Notes |
| --- | --- | --- |
| `BACKBONE` | `wide_resnet50_2` | |
| `LAYERS` | `["layer2", "layer3"]` | |
| `CORESET_RATIO` | `0.1` | lower = smaller bank, faster, less coverage |
| `NUM_NEIGHBORS` | `9` | |
| `VAL_SPLIT_RATIO` | `0.1` | good images held out to calibrate |
| `PATCHCORE_OVERRIDES` | `{}` | per-category overrides; intentionally empty |

### Classifier

| Setting | Default | Notes |
| --- | --- | --- |
| `RANDOM_SEED` | `42` | everything is seeded from this |
| `CLASSIFIER_FOLDS` | `5` | |
| `DEFAULT_PCA_COMPONENTS` | `40` | chosen by a measured sweep |
| `CLASSIFIER_C` | `0.2` | chosen by a measured sweep |
| `MIN_PER_CLASS_FOR_CV` | `5` | below this, CV is reduced or suppressed |
| `MIN_PER_CLASS_FOR_NESTED_CV` | `7` | below this, nested CV is skipped |

### Environment variables

| Variable | Purpose |
| --- | --- |
| `DEFECT_DETECTOR_DEVICE` | `cpu` or `cuda` — pin the serving device. Useful to run the API on CPU while the GPU trains. |
| `VITE_API_BASE` | Frontend's API base URL (default `http://127.0.0.1:8000`) |

---

## 11. File and data layout

```
defect-detector/
├── src/
│   ├── config.py            paths, hyper-parameters, CV guards
│   ├── dataset.py           folder enumeration, masks, split discovery
│   ├── imaging.py           preprocessing, heatmap, outline, data URIs
│   ├── patchcore_infer.py   standalone inference from a memory bank
│   ├── features.py          appearance + geometry features
│   ├── estimators.py        SafePCA (own module so the pickle is portable)
│   ├── train_patchcore.py   stage 1
│   ├── calibrate.py         stage 2
│   ├── train_classifier.py  stage 3
│   ├── compare.py           cross-category table
│   ├── run_all.py           multi-category orchestration
│   ├── pipeline.py          detect → localise → explain, multi-category serving
│   └── api/main.py          FastAPI app
├── scripts/
│   ├── train_all.ps1            whole pipeline
│   ├── onboard_new_category.py  add a new product
│   ├── recalibrate_all.py       re-derive thresholds only
│   ├── smoke_test.py            end-to-end check
│   ├── api_test.py              route check
│   ├── render_samples.py        visual montages
│   └── domain_shift_test.py     out-of-domain behaviour
├── models/<category>/
│   ├── patchcore.pt             memory bank + settings (110–150 MB)
│   └── defect_classifier.joblib absent if no labelled defects
├── outputs/
│   ├── comparison.json          cross-category table
│   └── <category>/
│       ├── metrics.json         thresholds, metrics, classifier block
│       ├── splits.json          which images went in the bank vs calibration
│       ├── classifier_features.npz  cached features
│       ├── category.json        onboarded categories only: type + source paths
│       └── sample_results.png   visual montage
├── data/onboarded/<category>/   staged client photos, MVTec layout
└── frontend/src/
    ├── App.tsx  App.css  api.ts
```

`models/`, `outputs/` and `data/onboarded/` are git-ignored — they are large and
regenerable.

### `metrics.json` schema

| Key | Meaning |
| --- | --- |
| `category`, `type`, `onboarded` | identity |
| `has_test_data`, `has_defect_labels` | what could be measured |
| `patchcore` | backbone, layers, coreset ratio, neighbours, bank shape |
| `image_auroc`, `image_average_precision` | threshold-free detection quality; `null` if unmeasurable |
| `pixel_auroc`, `pixel_average_precision`, `pixel_f1_at_threshold` | localisation quality; `null` without masks |
| `image_threshold`, `pixel_threshold` | the operating thresholds |
| `threshold_rule` | which rule produced `image_threshold` |
| `at_threshold` | accuracy/precision/recall/confusion at the default threshold |
| `at_max_threshold`, `max_threshold` | the alternative rule, for comparison |
| `calibration_scores` | min/median/mean/max/MAD/n of held-out good scores |
| `per_defect_recall` | detection recall per defect type |
| `counts` | bank / calibration / test image counts, defect class count |
| `heatmap_range`, `score_range` | display ranges for the UI |
| `classifier` | classes, counts, CV and nested-CV accuracy, confusion, status strings |

---

## 12. Interpreting the output

### 12.1 Anomaly score and threshold

The score is a **raw distance in embedding space**, not a probability. It is only
meaningful relative to that category's threshold, and thresholds differ widely
between categories (bottle 31.4, metal_nut 39.6, wood 33.8). Never compare a score
across categories.

### 12.2 Which accuracy number to quote

- **Image AUROC** is threshold-free and the best single measure of detection
  quality.
- **Detection accuracy** depends on the threshold and is the practical number.
- **Classifier accuracy**: quote the **nested CV** figure. Plain CV is mildly
  optimistic. With 57–109 samples both carry ±5–13%, so treat them as ranges.

### 12.3 `defect_area_pct` overstates

PatchCore scores a 32×32 grid, upsampled 8× and blurred (σ=4), which dilates the
region. A bottle chip covering ~6% of the frame reports nearer 12% (pixel recall
0.98, precision 0.46). The region reliably **contains** the defect but is not a
tight segmentation. On carpet and grid it is considerably worse (pixel F1
0.24–0.35). Use it as a relative severity signal, not a measurement.

### 12.4 Defect-type confidence is inflated in the demo

Each shipped classifier is refit on **all** of its category's labelled images,
including every `/samples` gallery entry, so it reports ~99% on those. That is
in-sample and meaningless as confidence. Probabilities are also **uncalibrated**
raw logistic outputs — no Platt or isotonic scaling — so even out-of-sample they
should not be read as literal likelihoods.

### 12.5 `null` versus zero

`null` means *not measurable*, never *zero*:

| Field | `null` when |
| --- | --- |
| `image_auroc` | the test split has only one class |
| `pixel_auroc`, `pixel_f1` | no ground-truth masks |
| `defect_type`, `defect_probabilities` | no classifier, or the part passed |
| `cv_accuracy_mean` | too few samples per class, or no labels at all |

---

## 13. Troubleshooting

**The UI shows an old set of categories.**
The React app fetches `/categories` once on mount. Refresh the page. If the
category is genuinely missing, restart the API — categories are discovered at
startup, not per request.

**`No trained categories found`.**
No `models/<name>/patchcore.pt` exists. Run the training pipeline or onboard a
category.

**`Missing outputs/<category>/metrics.json`.**
Calibration has not run for that category. Run `src.calibrate --category <name>`.

**CUDA out of memory during training.**
Lower `--batch-size`, or `--accelerator cpu`. Stop the API first — it holds memory
banks on the GPU.

**Coreset selection crawling at ~20 it/s.**
VRAM thrashing (§7.2). It will finish. Free GPU memory or use a larger card.

**`Access is denied` deleting a staged category.**
Read-only source media. The onboarding script handles this via `--force`; manually,
clear the read-only attribute first.

**Frontend can't reach the API.**
Check `GET /health`. If you serve the UI from another origin, add it to the CORS
list in `src/api/main.py`.

**High false alarm rate on a new category.**
Expected below ~200 good photos (§8.5). Supply more. Check `calibration_scores.n`
and `calibration_scores.mad` in `metrics.json`; a small `mad` with few samples means
the threshold is too tight.

**Predictions look wrong after renaming a category.**
A category name shadows a same-named MVTec folder. Check
`config.dataset_root_for("<name>")` and `outputs/<name>/splits.json` to see which
images were actually used.

---

## 14. Limits — what this system cannot do

### 14.1 It has no idea when it is out of its depth — measured

Every threshold is an absolute distance calibrated against one product on one rig.
`scripts/domain_shift_test.py` measured what that means:

| Detector | own good (control) | other categories' good | synthetic |
| --- | --- | --- | --- |
| every category | 0–1 of 5 flagged | **25/25 flagged** | **3/3 flagged** |

**100% of out-of-domain input is reported as defective**, at 2–3× the threshold,
with a confident defect-type label attached. Concretely: a **clean carpet photo**
shown to the bottle detector scores 67.7 against a 31.4 threshold and returns
`broken_small` at **100.0% confidence**.

This is not a bug — PatchCore reports distance to its own memory bank, and anything
unfamiliar is far away. But **the system cannot tell you "this is not something I
can assess."** Anything deployed in front of it needs an upstream check that the
image is the right product on the right rig. The model cannot be that check.

*(Genuine random internet images were never tested; the above is an offline proxy
using real photographs from other categories plus synthetic images.)*

### 14.2 Closed set of defect types

The classifier has no "unknown" class and no abstention threshold. A novel failure
mode will still be **detected** — that is the point of an unsupervised detector —
but will then be forced into one of that category's 3–5 known labels, usually with
high confidence. The score and heatmap stay trustworthy; **the type label does
not.**

### 14.3 Other limits

- **Thresholds are rig-specific.** Different camera, lighting or working distance
  requires re-calibration against that rig's good parts.
- **Aspect ratio is destroyed.** Everything is resized to a square 256×256 with no
  letterboxing. Fine for MVTec; a non-square input will be distorted.
- **No pixel metrics for onboarded categories** without client annotations.
- **Onboarding is manual.** No client-facing upload, no queue, no auth, no tenancy.
- **No check that `--good-dir` contains one product.** Point it at a mixed folder
  and it builds a memory bank spanning all of it, with a useless threshold.
- **Single seed.** Everything ran at `RANDOM_SEED=42`; the ± figures are
  cross-validation spread, not seed-to-seed spread.
- **The UI has never been opened in a browser by an automated check.** It compiles,
  builds, serves, and every API route it uses is tested — but the rendered page has
  not been visually verified programmatically.

See PROJECT_REPORT §8 for the full list of outstanding work.
