# V1.6 algorithm audit and comparison

No real labeled recipe footage, Blackmagic SDK, or CLIP weights were available here. **Real-world grouping accuracy and CLIP inference performance remain unmeasured.** The supplied manifests contain explicitly handcrafted vectors. They exercise decision rules; they are not embeddings from a model, demo images, or BRAW footage. The user’s observation that CLIP left more clips Uncertain than visual features is not a controlled comparison.

## Findings in the existing pipeline

- `EmbeddingCache` opens cached samples as RGB. CLIP uses its own local `CLIPProcessor`, evaluation mode, inference mode and normalized image features. Both fresh/cached vectors pass finite/nonzero/dimension/count checks. Norm overflow is now rejected too. No evidence was found of a missing-normalization or RGB-conversion bug causing the reported uncertainty.
- Cache keys include backend, model fingerprint/version, source fingerprint, sampling version, and ordered frame IDs. CLIP and features have separate cached vectors. A CLIP request never substitutes features. Existing cached CLIP vectors can be compared without loading Torch; missing/stale vectors require local weights. The model fingerprint uses model-file names/sizes/mtimes, not a full weights hash; preserve file timestamps honestly or clear/rebuild after unusual manual model edits.
- Automatic pairing combines bidirectional best-frame coverage (65%) and clip-centroid similarity (35%). Complete linkage requires every cross-cluster pair to clear its visual gate. Multi-stage recipes can fragment even with strong intermediate matches. Model-specific defaults are features .89 / CLIP .88 / historical hybrid .885; these are heuristics, not professionally calibrated thresholds. Genuine recording timestamps add only weak evidence; filesystem mtime is never used as recording time.
- Guided V1.5 matching already uses an example bank, with separate lower review gates (.90 features / .78 CLIP / .80 hybrid), coverage and competing-group margins. It does not require all examples to resemble each other. However, more example frames create more opportunities for accidental nearest-neighbor matches, and repeated views can overstate support.
- Manually assigned clips, including manually assigned Uncertain, are protected and excluded from recommendation candidates. This can explain apparent ineligibility, but not every Uncertain result. Reset now removes those active overrides, and summaries count manual protection separately.
- Empty-group accumulation had a concrete storage cause: candidate groups were published even when no effective member remained after manual overrides/uncertainty. V1.6 only publishes groups with effective automatic members and retires obsolete empty automatic groups, retaining historical references and empty user targets.

The persisted analysis summary includes processed/failed/protected/grouped/uncertain/unsorted counts, reason counts and pairwise visual-score quantiles. Clip evidence retains the top five neighbors, best/second candidate IDs where a live group exists, raw cluster IDs and reasons. Candidate evidence does not create an empty group just to display a name. These diagnostics allow a workstation dataset to distinguish weak evidence, competing groups, fragmentation and manual protection before choosing thresholds.

## Selectable experiments

**Guarded average-link:** merge on mean cross-group visual similarity, with a minimum pairwise floor 0.14 below the threshold. An accepted member must retain a strong companion, adequate average/floor and separation from rival groups. It can accommodate moderate stage variation while preventing unrestricted single-link chaining. Heap/matrix implementation remains quadratic in pair storage and retains the 1,500-clip library cap. It cannot reconstruct arbitrarily different stages without connecting visual evidence. Complete linkage stays the default.

**Balanced diverse prototypes:** normalize and collapse near-duplicate example views, choose deterministic farthest-point representatives (at most eight), then give each competing recipe the same prototype budget, bounded by the smallest recipe's distinct examples. Candidate duplicate views also collapse for aggregation. Require supporting coverage, per-frame rival separation, overall margin and at least two distinct supporting prototypes for the “suggested” label. One repeated background cannot alone satisfy that final test. Fewer prototypes in a poorly represented rival can limit every group; collecting diverse examples remains important. No competing examples is explicitly reported. All candidates, including ambiguous ones, remain reviewable; nothing moves until accepted. Automatic proposals never become trusted examples implicitly; run examples are frozen.

**Hybrid:** V1.5's 75/25 raw CLIP/features combination remains a labeled historical control, not a justified universal scale conversion. The new prototype hybrid uses CLIP ranking and a features-based competing-group veto (> .04 lead), rather than mixing raw scores. That veto is itself experimental and not calibrated on real footage. Synthetic results do not justify making hybrid the default.

**Sampling:** uniform uses five positions with bounded quality retries. Existing adaptive sampling uses that base plus up to four change-driven midpoints. New diverse sampling reuses the same bounded pool, preserves early/middle/late anchors and keeps up to seven diverse frames (normally at least five). It is not shot detection or a full-video scan. With quality retries, “five frames” can require up to 15 decode attempts, adaptive/diverse up to 19. Changing strategies reuses existing valid JPEG samples at matching positions; a test verifies base BRAW-shaped frame calls are not repeated. Only the active strategy's frames/embedding records are retained, so export each comparison manifest before switching strategies. No unconditional quality improvement is claimed.

## Metrics and split policy

A manifest assigns each whole clip to example, calibration, or evaluation. Frames never cross splits. Guided inference sees only example labels; calibration tunes thresholds on calibration outcomes; evaluation labels are only consulted for metrics. Automatic clustering includes example clips as **unlabeled visual context** and held-out candidate clips; it never sees manual assignments or recipe labels. Calibration clips are absent from evaluation clustering. Metrics apply only to held-out candidates. Adding hidden truth seeds or counting manual corrections as automatic successes is forbidden by construction.

With at least three fully labeled filming sessions, export partitions whole sessions deterministically. Otherwise it falls back to clip-stratified splits with an explicit leakage warning (at least five labeled clips per recipe). Explicit splits require a split on every labeled clip. A session spanning splits emits a warning. Recipe/session distribution may leave a recipe without example coverage; that failure is retained rather than silently adding seeds. Source-fingerprint changes invalidate truth export until reviewed.

- Pair precision = correctly grouped same-recipe pairs / all pairs placed together. Pair recall = recovered same-recipe pairs / all true same-recipe pairs, including abstentions. No predicted pairs gives undefined precision (`null`), not 100%.
- Coverage = assigned evaluation clips / all evaluation clips; uncertainty = the complement. Uncertain is abstention, never one successful recipe cluster.
- Contamination = assigned clips in any mixed-recipe predicted group / assigned clips. This is a strict impurity measure, not majority-label error.
- Fragmentation = mean predicted component count per recipe with at least one assignment. Per-recipe counts (zero when fully abstained) and fully abstained recipe counts are also reported.
- Suggested-match accuracy = correct named-group suggestions / suggestions labeled “suggested.” The ungated top-ranked accuracy is reported separately. Automatic group IDs are not interpreted as recipe names.
- Quality@10 = correct / reviewed among the first min(10, candidate count) results for **each recipe queue**, including ambiguous candidates. The report preserves per-queue and pooled denominators. A clip reviewed in two queues is counted twice.
- Runtime is scoring only, excluding calibration search and embedding preparation. Preparation has its own timing/cache statistics. Python peak allocations exclude some native memory; RSS is the cumulative process high-water mark in KiB, not per-method GPU/total peak. Frozen embedding array sizes are recorded.

Default and calibrated operating points are reported separately. Calibration seeks maximum coverage subject to 95% pair precision for automatic clustering, or 95% named-group precision for guided suggestions. If no gate qualifies, calibrated output abstains instead of relaxing the error requirement. Pair precision alone can hide singleton errors: inspect contamination, fragmentation and recommendation metrics as well. Tiny synthetic samples cannot establish a reliable error rate.

## Executed results (handcrafted vectors only)

`benchmarks/v16-stage-*` constructs two recipes at 0/18/36-degree stages with ±5-degree within-clip variation and an unseen pan. This is an intentionally selected regression boundary, not an unbiased accuracy dataset. Four seeds, seven calibration clips and seven evaluation clips are used. CLIP/features slots intentionally contain identical analytical vectors, so differences between their rows reflect thresholds rather than real model behavior.

At the default CLIP threshold, complete-link coverage is 4/7, recall 0, and fragmentation 2. Guarded average-link achieves coverage 6/7, precision/recall 1, and fragmentation 1, with the unseen pan abstaining. **Calibration also recovers 6/7 with precision/recall 1 for complete linkage**; this fixture establishes neither general superiority nor the root cause of the user's observation. Both guided methods cover 6/7 here. Quality@10 is 6/14 (42.9%) because each of the two recipe queues contains all seven candidates; accepted-match accuracy is a different denominator.

`benchmarks/v16-lookalike-*` preserves the V1.5 stress fixture: orthogonal stages, competing similar bowls and an unseen pan. Revised automatic grouping cannot repair misleading hypothetical CLIP vectors: its default result, like complete linkage, has zero pair precision and 50% contamination among assigned clips. Precision-constrained calibration abstains completely. Prototype safeguards label fewer candidates suggested than the old bank, keeping repetitive single-view candidates ambiguous. That is a measured coverage cost, not an accuracy breakthrough. Baseline and hybrid differences here are engineered into the vectors, not measured model behavior.

### Stage fixture: default operating points

| Backend | Method | Pair precision | Pair recall | Coverage | Contamination | Suggested accuracy |
|---|---|---:|---:|---:|---:|---:|
| features | complete | undefined | 0.0% | 57.1% | 0.0% | undefined |
| features | guarded | undefined | 0.0% | 57.1% | 0.0% | undefined |
| features | bank | 100.0% | 100.0% | 85.7% | 0.0% | 100.0% |
| features | prototypes | 100.0% | 100.0% | 85.7% | 0.0% | 100.0% |
| clip | complete | undefined | 0.0% | 57.1% | 0.0% | undefined |
| clip | guarded | 100.0% | 100.0% | 85.7% | 0.0% | undefined |
| clip | bank | 100.0% | 100.0% | 85.7% | 0.0% | 100.0% |
| clip | prototypes | 100.0% | 100.0% | 85.7% | 0.0% | 100.0% |
| hybrid | complete | undefined | 0.0% | 57.1% | 0.0% | undefined |
| hybrid | guarded | undefined | 0.0% | 57.1% | 0.0% | undefined |
| hybrid | bank | 100.0% | 100.0% | 85.7% | 0.0% | 100.0% |
| hybrid | prototypes | 100.0% | 100.0% | 85.7% | 0.0% | 100.0% |

Scoring across the 24 rows: 0.011–0.031 seconds. Maximum cumulative RSS: 27996 KiB. These tiny-vector timings are not CLIP inference or production sizing. Full calibrated rows/denominators are in the JSON report.

### Lookalike fixture: default operating points

| Backend | Method | Pair precision | Pair recall | Coverage | Contamination | Suggested accuracy |
|---|---|---:|---:|---:|---:|---:|
| features | complete | undefined | 0.0% | 57.1% | 0.0% | undefined |
| features | guarded | undefined | 0.0% | 57.1% | 0.0% | undefined |
| features | bank | 100.0% | 100.0% | 85.7% | 0.0% | 100.0% |
| features | prototypes | undefined | 0.0% | 28.6% | 0.0% | 100.0% |
| clip | complete | 0.0% | 0.0% | 57.1% | 50.0% | undefined |
| clip | guarded | 0.0% | 0.0% | 57.1% | 50.0% | undefined |
| clip | bank | 100.0% | 33.3% | 57.1% | 0.0% | 100.0% |
| clip | prototypes | undefined | 0.0% | 28.6% | 0.0% | 100.0% |
| hybrid | complete | undefined | 0.0% | 28.6% | 0.0% | undefined |
| hybrid | guarded | undefined | 0.0% | 28.6% | 0.0% | undefined |
| hybrid | bank | 100.0% | 33.3% | 57.1% | 0.0% | 100.0% |
| hybrid | prototypes | undefined | 0.0% | 28.6% | 0.0% | 100.0% |

Scoring across the 24 rows: 0.011–0.038 seconds. Maximum cumulative RSS: 28124 KiB. These tiny-vector timings are not CLIP inference or production sizing. Full calibrated rows/denominators are in the JSON report.

### Executed sampling comparison

A real FFmpeg-decoded, read-only 4-second 320×180 `testsrc2` MP4 was sampled. Uniform retained 5 frames / 5 decode calls in 0.761 s; adaptive retained 6 / 6 in 0.870 s; diverse retained 6 / 6 in 0.759 s. Adaptive and diverse chose the same times on this fixture. These single-run timings include normal noise and do not show a diverse-sampling speed advantage. SHA-256, size and nanosecond mtime remained unchanged. See `benchmarks/v16-sampling-pattern.json`; this measures decoding mechanics, not recipe accuracy or BRAW performance.

## Reproduce and evaluate your own footage

Stop the server before CLI analysis. Use the clip-detail **Independent benchmark truth** fields to label known recipes, filming session IDs and stages. Prefer recipe identities that remain consistent across sessions. Optional explicit splits are example/calibration/evaluation. Alternatively, after verifying manual groups, explicitly snapshot them with `benchmark-labels --from-groups`; this overwrites labels on those manually grouped clips, so use the UI if you already curated session/split information. Reset sorting never erases benchmark labels.

```bash
# Existing application data directory, safely separate from source footage:
recipe-sorter --data-dir "$HOME/.local/share/recipe-clip-sorter" benchmark-labels --from-groups
# Export fixed truth and both local embedding backends:
recipe-sorter --data-dir "$HOME/.local/share/recipe-clip-sorter" benchmark-export --prepare --backends clip features --seed 17 --output benchmarks/uniform-labels.json
recipe-sorter --data-dir "$HOME/.local/share/recipe-clip-sorter" benchmark --suite v16 --manifest "$HOME/.local/share/recipe-clip-sorter/benchmarks/uniform-labels.json" --output benchmarks/uniform-results.json
# Same truth/split seed, optional bounded sampling comparisons:
recipe-sorter --data-dir "$HOME/.local/share/recipe-clip-sorter" analyze --backend clip --sampling adaptive --method guarded
recipe-sorter --data-dir "$HOME/.local/share/recipe-clip-sorter" benchmark-export --prepare --seed 17 --output benchmarks/adaptive-labels.json
recipe-sorter --data-dir "$HOME/.local/share/recipe-clip-sorter" benchmark --suite v16 --manifest "$HOME/.local/share/recipe-clip-sorter/benchmarks/adaptive-labels.json" --output benchmarks/adaptive-results.json
recipe-sorter --data-dir "$HOME/.local/share/recipe-clip-sorter" analyze --backend clip --sampling diverse --method guarded
recipe-sorter --data-dir "$HOME/.local/share/recipe-clip-sorter" benchmark-export --prepare --seed 17 --output benchmarks/diverse-labels.json
recipe-sorter --data-dir "$HOME/.local/share/recipe-clip-sorter" benchmark --suite v16 --manifest "$HOME/.local/share/recipe-clip-sorter/benchmarks/diverse-labels.json" --output benchmarks/diverse-results.json
# Independent single-clip sampling cost; works with the existing BRAW SDK bridge:
python scripts/sampling_benchmark.py /path/to/clip.braw --data-dir /tmp/rcs-sampling
# Analytical regression suite, no model or footage download:
recipe-sorter --data-dir /tmp/rcs-benchmark benchmark --suite v16 --synthetic
recipe-sorter --data-dir /tmp/rcs-benchmark benchmark --suite v16 --manifest benchmarks/v16-lookalike-labels.json --output benchmarks/lookalike-results.json
# Preserve the old V1.5 scoring controls:
recipe-sorter --data-dir /tmp/rcs-benchmark benchmark --suite v15 --synthetic --output benchmarks/v15-results.json
```

Before the first export, run uniform analysis if the library currently uses a different sampler. Keep the exported manifests fixed for algorithm comparisons. The commands write reports only inside their application data directory. `--prepare` can decode missing frames and compute missing local embeddings; without it, export reads only valid caches and the benchmark explicitly skips incomplete backends. Reports can contain filenames and human labels; inspect them before sharing.

Remaining evaluation needs: actual CLIP/baseline score distributions; session-held-out recipe videos with competing bowls/backgrounds and different stages; human color/exposure inspection; model inference/RAW proxy timing; larger-library memory and desktop browser behavior. Keep experimental methods selectable until those results justify a default change. Static image matching does not imply ingredient-transfer, action or temporal understanding.
