# CLIP audit and reproducible evaluation

The actual user's footage, local CLIP weights/PyTorch, Fedora workstation, and BRAW SDK were not available. The measurements below exercise **handcrafted vector fixtures**, not CLIP inference or professional recipe-group accuracy. Real model conclusions require the labeled workflow below.

## Why CLIP can leave more clips Uncertain

The code audit found no missing L2 normalization in V1: model output frames and clip centroids were normalized, and frame comparison used cosine dot products. V1's behavior has several other causes worth measuring on real footage:

1. **Different score distributions.** The color/layout/texture descriptor is largely nonnegative and can give high similarity to clips sharing a bowl, counter or lighting. CLIP's learned directions have a different distribution. V1's .89 baseline and .88 CLIP gates were uncalibrated. Fewer Uncertain clips from the baseline does not demonstrate greater correctness.
2. **Complete-link consistency.** V1 requires every member to clear a pairwise visual gate against every other member. Butter mixing, baking and a finished cake need not look mutually alike. A valid recipe thread can consequently fragment. Lowering the gate can merge unrelated bowls/pans instead of recovering the thread.
3. **Centroid and symmetric coverage.** V1 uses 65% symmetric nearest-frame coverage plus 35% centroid similarity. A short clip of one stage can score poorly against a longer clip spanning several stages; the centroid can average away the most useful visual evidence.
4. **Five fixed samples.** These are efficient and avoid first-frame bias, but may miss a short distinctive action. Repeated generic frames are not independent evidence about recipe identity.
5. **Cache coupling.** V1 stores one embedding per frame and ties reuse of frames to the selected model version. Changing the backend can replace the previous vectors and unnecessarily decode source frames again. This costs time; it is not evidence of a normalization defect.

V1.5 keeps the old baseline/complete-link algorithm and thresholds for comparison, separates frame and model caches, rejects nonfinite/zero/dimension-invalid vectors, checks vector/frame alignment, and re-normalizes cached vectors at the scoring boundary. Cache identity includes source fingerprint, sampling version, ordered frame IDs and actual model version. A recommendation run refuses mixed model generations. Existing CLIP vectors can be compared without loading PyTorch; missing vectors still require the real local model, never a substitute.

## Example-bank scoring

Each group is represented by the frames of its manually assigned example clips. Every candidate frame finds its highest cosine similarity to **any** example frame. A frame showing creaming may match one example while a baking frame matches another. Examples are not required to match each other.

The group score is `0.8 × mean(nearest similarities) + 0.2 × lower quartile(nearest similarities)`. This reduces the influence of a single matching frame. Review triage also requires enough supported frames and a lead over competing example groups. Defaults are deliberately backend-specific, but are **not calibrated on this user's media**:

| Backend | Group threshold | Lead over rival | Per-frame support floor | Required frame coverage |
| --- | ---: | ---: | ---: | ---: |
| CLIP ViT-B/32 | .78 | .040 | .74 | 60% |
| Visual features | .90 | .035 | .86 | 60% |
| Experimental hybrid | .80 | .040 | .76 | 60% |

Hybrid combines 75% CLIP bank similarity and 25% baseline bank similarity. It additionally requires the CLIP component to reach .74; appearance alone cannot bypass weak semantic evidence. It uses exactly the same sampled frame positions for both backends. The strongest CLIP example-frame pairs are displayed as support; component scores remain in the stored evidence.

Thresholds label the review queue as Suggested match or Needs your judgment. They do not move clips. The human can accept an ambiguous candidate after inspecting it. No score is presented as a probability. Recording time does not influence the example-bank method; the preserved V1 control still has its small, bounded metadata bonuses.

This method has limitations: too many broad or incorrect examples can match unrelated footage; a clip with generic bowls throughout may still be ambiguous; missing competitor examples remove the separation check. It reconstructs visual associations, not causal food-state progression. Accepted/rejected decisions are retained but do not fine-tune CLIP or train a negative classifier.

## Adaptive sampling experiment

Uniform sampling remains the default. Adaptive sampling first uses the same five quality-filtered samples, then compares adjacent frames using cheap visual descriptors. Adjacent similarity below .94 requests a midpoint sample. This adds at most four frames, for at most nine retained images. Existing black/blur quality gates apply; unreadable extra samples are skipped. Base sampling may make nearby retry attempts, so the decode-call count can exceed the retained-frame count.

The deterministic sampler test detects a mocked red-to-blue visual transition, retains all five uniform positions, adds a useful midpoint, and stays within nine samples. This verifies mechanics, not recipe-stage recall. The sampling experiment can miss a brief action between visually similar endpoints or sample camera motion. It has not demonstrated an improvement on actual CLIP footage. Changing strategies invalidates derived embeddings and old review evidence, while manual assignments persist.

## Synthetic regression results

Reproduce from the project root:

```bash
python -m recipe_sorter --data-dir /tmp/rcs-v15-eval benchmark --synthetic
# Or use the exact supplied manifest:
python -m recipe_sorter --data-dir /tmp/rcs-v15-eval benchmark \
  --manifest benchmarks/synthetic-labels.json
```

The manifest contains 18 analytical examples: four seed clips (mixing and baked stages for each of two recipes), seven calibration clips and seven evaluation clips. Each evaluation/calibration split has transitions between stages, finished stages, visually similar bowls from different recipes, and an unknown pan. The calibration and evaluation distributions are intentionally almost identical; this is a small deterministic regression fixture, not an independent validation study. Held-out evaluation labels do not enter calibration or scoring.

Calibration maximizes the number assigned while meeting 95% suggested-match accuracy on calibration clips; conservative threshold/margin values break ties. If no setting meets the target, it abstains entirely. The fixture selected .96 threshold/.12 margin for all three banks. These selected settings apply to **this benchmark only**, not automatically to the application.

| Descriptor fixture / method | Pair precision | Pair recall | Uncertain | Suggested-match accuracy |
| --- | ---: | ---: | ---: | ---: |
| Visual / V1 complete-link | undefined | 0% | 42.9% | n/a |
| Visual / example bank | 100% | 100% | 14.3% | 100% (6/6) |
| CLIP-like / V1 complete-link | 0% | 0% | 42.9% | n/a |
| CLIP-like / example bank | 100% | 33.3% | 42.9% | 100% (4/4) |
| Hybrid / V1 complete-link | undefined | 0% | 71.4% | n/a |
| Hybrid / example bank | 100% | 33.3% | 42.9% | 100% (4/4) |

“Undefined” means there are no predicted pairs among the evaluation clips, not perfect precision. The original algorithm may cluster an evaluation clip with a seed; only evaluation-to-evaluation pairs count in these metrics. It receives no recipe labels, while the example bank intentionally receives seed labels. The comparison measures the benefit of human examples, not equivalent supervision.

In this constructed counterexample, hybrid improves top-ranked accuracy from CLIP-like 4/7 to 6/7 while retaining the same conservative four accepted candidates and Uncertain rate. A separate math test demonstrates a case where visual corroboration fixes a semantic lookalike; another ensures visual features cannot bypass the CLIP floor. This justifies an **optional experimental comparison**, not making hybrid the default or claiming better real accuracy.

Frozen scoring took approximately 10–21 ms per method on this tiny fixture, with roughly 20–32 KiB peak traced Python allocations and 28–29 MiB cumulative process RSS. These exclude model loading, GPU allocation, decoding, calibration time, and real-world descriptor dimensions. They are not workstation capacity estimates. Exact run records, algorithm versions, manifest hash, profiles and measurements are in [benchmarks/synthetic-results.json](benchmarks/synthetic-results.json).

## Benchmark your manually labeled footage

Use a separate application library per evaluation session. Analyze normally, then manually label **at least five clips per recipe**, preferably many more spanning preparation, mixing, cooking and finished stages. Include similar bowls/actions from different recipes, ambiguous clips with known labels, and multiple filming sessions/cameras. Do not label based solely on the app's suggestions. Have a second person check a subset where practical.

Stop the server before these CLI commands. Replace `/path/to/app-data` with your app-owned library, never the footage directory:

```bash
# One-time local model setup, if not already installed in this library:
python -m pip install -e '.[ai]'
recipe-sorter --data-dir /path/to/app-data prepare-model

# Build/reuse both embeddings without replacing manual labels/proposals.
recipe-sorter --data-dir /path/to/app-data benchmark-export --prepare \
  --backends clip features --seed 17 --output benchmarks/uniform-labels.json
recipe-sorter --data-dir /path/to/app-data benchmark \
  --manifest /path/to/app-data/benchmarks/uniform-labels.json \
  --target-precision .95 --output benchmarks/uniform-results.json
```

Without `--prepare`, export reads only current valid cached embeddings, never loads a model or decodes a source. Missing or stale backends are omitted and explicitly skipped by the benchmark. It does not silently reuse old model/sampling generations or fabricate CLIP data. `--prepare` reports extraction/reuse time, cache counts and process RSS separately. It preserves the current sampling strategy per clip, so run uniform analysis first if your library contains mixed strategies.

The exporter splits clips deterministically within each recipe using the seed: approximately one third examples (minimum two), one quarter calibration (minimum one), and the remainder evaluation. The manifest includes fingerprints, sampling/model versions and frame vectors; keep it for reproducibility. It does not contain source media, but filenames/labels may still be private. All generated outputs are constrained to app data.

For a stronger evaluation, edit the **exported manifest** to set `split` by filming session/sequence rather than random adjacent clips. Choose seed clips across recipe stages; reserve different sessions for calibration/evaluation where possible. Add held-out recipe labels without seed examples to measure unknown-recipe abstention. Clips manually in Uncertain are not exported by default because Uncertain is not ground truth. The same clip ID cannot occur in two splits; different files from the same scene can still leak visual content, so inspect split membership.

To compare sampling while keeping labels and the deterministic split fixed:

```bash
recipe-sorter --data-dir /path/to/app-data analyze --backend clip --sampling adaptive
recipe-sorter --data-dir /path/to/app-data benchmark-export --prepare \
  --backends clip features --seed 17 --output benchmarks/adaptive-labels.json
recipe-sorter --data-dir /path/to/app-data benchmark \
  --manifest /path/to/app-data/benchmarks/adaptive-labels.json \
  --output benchmarks/adaptive-results.json
# Restore the five-frame default if preferred:
recipe-sorter --data-dir /path/to/app-data analyze --backend clip --sampling uniform
```

Each report compares CLIP, visual features and hybrid under both complete-link and example-bank methods. Metrics:

- **Pair precision:** correct predicted same-recipe pairs / all predicted pairs.
- **Pair recall:** recovered true pairs / all true same-recipe pairs, including pairs missed through abstention.
- **Uncertain fraction:** unassigned evaluation clips / all evaluation clips.
- **Suggested-match accuracy:** correct named-group assignments / suggested assignments; undefined if all abstain. In a benchmark these simulate the suggested label, not a real human acceptance.
- **Top-ranked accuracy:** correct best group / all evaluation clips before threshold abstention. An unknown recipe without seeds counts as incorrect here even if correctly left Uncertain by the gate.
- **Scoring wall time, traced Python allocation peak, cumulative Linux process RSS high-water mark, and frame-array storage bytes.** NumPy/native/Torch memory is not fully measured by tracemalloc; CUDA memory needs separate workstation profiling. Export preparation reports cover frame/model work separately; record cold and warm runs explicitly.

Compare precision first, then coverage/recall. Never pick a threshold on the evaluation labels. Calibration on a tiny sample cannot establish a reliable 95% true accuracy guarantee. Report sample sizes, recipe/session composition, hardware, Fedora and SDK versions, model weights, cold/warm cache state, and human errors along with the numbers. Real performance and accuracy remain open until these commands run against labeled footage.
