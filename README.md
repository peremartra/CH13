# CH13

This chapter builds a small **family of function-calling models** through a
cascade of fine-tuning, structured pruning, and knowledge distillation, all
based on Qwen3. Starting from a general-purpose base model, we specialize it
into a weather/geolocation tool-calling expert (**T**), and then shrink it
twice — **T → M → S** — using each step's output as the teacher and pruning
source for the next. Every notebook applies the same core procedure (curate
data → prune → distill → evaluate with tolerant matching and multi-seed
variance checks), so the results are directly comparable across the family,
and the differences between notebooks come from what gets pruned and from
which teacher, rather than from a change in method.

The notebooks are meant to be run in this order:

1. `CH13_NB01_LoRA_weather_specialist.ipynb` — creates the teacher, **T**.
2. `CH13_family_model_M_v1.ipynb` — prunes and distills **T → M**.
3. `CH13_family_model_S_v1.ipynb` — prunes and distills **M → S**.

## 1. `CH13_NB01_LoRA_weather_specialist.ipynb`

Fine-tunes `Qwen/Qwen3-0.6B` with LoRA on a curated subset of the
`Salesforce/xlam-function-calling-60k` dataset, producing `specialist_model`
(**T**), a weather/geolocation tool-calling expert used as the teacher for the
rest of the chapter.

What it does:
- **Curates the domain data**: keeps only examples that call one of three
  tools (`get_ip_zipcode`, `get_city_from_zipcode`, `local_weather_api`),
  deduplicates near-identical queries (xlam is synthetically generated and
  repeats itself), and filters out examples that require world knowledge the
  query never states, so the task becomes pure extraction rather than recall.
- **Builds prompt/completion pairs** from the chat template, isolating the
  assistant's `tool_call` so training can use `completion_only_loss=True`
  instead of wasting gradient on the auto-generated tool schema.
- **Trains a LoRA adapter** (rank 16, targeting all attention and MLP
  projections) on top of the fp16 base model, with a sanity check that the
  loss mask is actually applied before spending time on training, then merges
  the adapter back into the base weights.
- **Evaluates and diagnoses the result**: compares the base model against the
  fine-tuned specialist with exact and "tolerant" matching (numeric/case
  equivalence, coordinate tolerance, substring matches), checks for
  overfitting by scoring a sample of the training set, and buckets every
  mismatch into "no call / wrong function / wrong arguments" categories.

**Result:** `specialist_model` — a merged, fp16 Qwen3-0.6B checkpoint
(optionally pushed to `oopere/qwen3-0.6b-weather-geo-specialist`) that is the
common teacher (**T**) for both pruning notebooks that follow.

## 2. `CH13_family_model_M_v1.ipynb`

Derives **Model M** from the teacher **T** by applying depth pruning, then
width pruning, then knowledge distillation, using the `optipfair` library.

What it does:
- **Re-establishes the data split and evaluation harness** used for T, and
  loads T as a frozen teacher, reporting its parameter count and baseline
  accuracy as the reference point for everything that follows.
- **Depth-prunes T from 28 to 25 layers**: ranks layers by cosine distance
  between each block's input and output, then removes the three least
  important ones under two constraints — no edge layers, no adjacent pairs —
  rather than dropping one contiguous block.
- **Width-prunes the MLPs from 3072 to 2304** (a 225% expansion rate over the
  1024 hidden size): uses PPM/MAW neuron importance in "hybrid" mode, which
  weighs `down_proj` by activation statistics gathered from a calibration
  dataloader, protecting the first and last two MLPs at full width.
- **Distills T → M** with completion-only masked labels and a combined
  hard-label + KL-divergence loss, repeated over three seeds (42, 43, 44) to
  separate a real effect from batch-order noise; the canonical seed's model
  is fixed as **M** before looking at the scores, to avoid selecting on the
  test set.
- **Reports and saves results**: a results table across the pipeline stages
  (teacher → depth-only → depth+width → post-KD), parameter counts (including
  the non-embedding share, since the tied embedding table never shrinks), and
  a `model_m_handoff.json` file carrying the layer-importance maps and data
  split forward to the S notebook.

**Result:** `model_m` — a 25-layer, ~344M non-embedding-parameter Qwen3
checkpoint distilled from T (optionally pushed to
`oopere/qwen3-0.6b-weather-geo-M`), plus `model_m_handoff.json` for the next
stage.

## 3. `CH13_family_model_S_v1.ipynb`

Repeats the same procedure one step further down the cascade: **Model S** is
pruned from **M** (not from T) and distilled with **M** as its teacher, so any
capability M lost is not recoverable here.

What it does:
- **Loads M and T** (M as the pruning source, T only as a fixed reference
  point) and recomputes calibration statistics and layer importance from
  scratch on M's topology — M's ranking is not simply T's ranking minus three
  entries, since removing layers redistributes redundancy among survivors.
- **Depth-prunes M from 25 to 22 layers** using the same
  no-edge/no-adjacent selection rule, and composes an index map that traces
  every surviving layer in S back through M to its original position in T.
- **Width-prunes the MLPs toward ~1792** (`EXPANSION_RATE=175`, resolved
  relative to M's current 2304, not to the hidden size), again with
  calibration-driven PPM/MAW neuron selection, then checks what width
  actually landed before proceeding.
- **Distills M → S** with the same masked hard-label + KL loss, again across
  three seeds, and inspects the training loss curve for a collapsing KL term
  (a sign the student has converged close enough to the teacher for the loss
  to be numerical noise).
- **Reports results in depth**: a per-stage comparison table (M → depth-only
  → depth+width pre-KD → S post-KD), per-function accuracy, an explicit check
  for queries where S fails but M still succeeded (errors accumulating down
  the cascade), bootstrap confidence intervals on exact-match scores, and a
  family-wide summary table (layers, intermediate size, parameter counts,
  embedding share) comparing M and S.
- **Saves the model and handoff file** (`model_s_handoff.json`) with the
  full S→M→T index mapping and per-seed scores.

**Result:** `model_s` — a 22-layer, ~275M non-embedding-parameter Qwen3
checkpoint distilled from M (optionally pushed to
`oopere/qwen3-0.6b-weather-geo-S`), completing the T → M → S model family.