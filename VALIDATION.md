# V1.6 validation record

Development environment: restricted Debian Cloud sandbox, Python 3.12, FFmpeg/ffprobe 7.1.5. This is not Fedora acceptance. All development used isolated application directories and fixtures, never the user's library.

## Executed

- **77 Python tests passed** (`python -m unittest discover -s tests -v`, 13.952 seconds), including all 51 prior tests, with expected schema-version and explicit-confirmation updates. New tests cover populated non-demo organization, atomic reset/restore, real SQLite backups and integrity checks, backup-path failure, injected mid-transaction failure, concurrent resets, busy jobs, stale generations, inactive old undo/suggestions/rejections, restart, offline sources, independent truth, effective full-library cleanup and no empty-auto-group accumulation.
- Manual direct/bulk assignments, same-group no-op, persistent undo, reanalysis protection, source hashes/size/mtime, frame/embedding/source-record retention, generation-aware restoration and subsequent-work conflicts passed.
- **15 JavaScript interaction unit tests passed** with `node tests/frontend.test.cjs`, including dedicated drag routing, multi-selection/hidden-count ghost, cancelled/external/deleted-target drops, same-group no-op, accessible explicit confirmation, undo, failed save and failed view refresh. These use fake event targets and are not rendered-browser acceptance.
- Actual in-process ASGI requests exercised reset confirmation, counts, stale-token refusal, restoration, independent labels, existing local-only security and original API workflow. No network socket was required.
- Both 24-row analytical comparison suites executed: stage variation and competing lookalikes, all three embedding slots, both automatic and both guided methods, default and calibrated operating points. CLI `benchmark --suite v16 --synthetic` also executed. Frozen manifests/results and complete metric definitions are included. They contain handcrafted vectors, not CLIP inference.
- A real FFmpeg MP4 sampling comparison executed for uniform/adaptive/diverse. Source SHA-256, size and nanosecond mtime were unchanged. The existing real MP4 decode/proxy/cache/source-safety integration test passed.
- Sampling changes reuse valid cached base frames; model/provenance validation, independent session splits, stale truth during preparation, metric abstention/fragmentation/contamination, review@10 denominators, no hidden manual/truth seeds, stage joining and anti-chaining safeguards passed.
- Python compilation and JavaScript syntax checks passed. The BRAW adapter, native bridge source and build script were compared with the original V1.5 ZIP: all three are byte-identical.

## Blocked or not measured

**Rendered browser:** `python scripts/browser_acceptance.py` was attempted after fixing its source-checkout import path. It stops at `socket.socket()` with `PermissionError: [Errno 1] Operation not permitted`, before launching a server/browser. The existing `browser_smoke.py` has the same socket restriction. Neither rendered workflow passed here. Fake DOM and ASGI tests do not establish desktop dragging. The new real-browser script includes populated non-demo footage, actual drag/scroll/filter interactions, restart, cleanup, reset, backup-path failure, stale confirmation, stale actions, restoration and fresh recommendations. Run both scripts on Fedora; see V1_6.md.

**Fedora/BRAW:** the user reports the previous version runs and decodes their BRAW footage. This release preserves the integration; official SDK compilation, real decoding/color/proxy/source integrity were not revalidated in Cloud. Existing `braw-acceptance` remains the workstation check.

**Real CLIP/performance:** weights/inference/GPU behavior and labeled professional recipe accuracy remain unmeasured. Synthetic measurements do not validate production accuracy. Production-scale 1,500-clip memory/runtime, long BRAW proxies, browser-specific native drag behavior and color-managed footage need workstation evaluation. Experimental algorithms remain selectable and do not replace the defaults.

## Historical V1.5 record

The following records the previous release, not new V1.6 validation.

# V1.5 validation record

Development environment: Debian 13 Cloud sandbox, Python 3.12.14, FFmpeg/ffprobe 7.1.5. This is **not Fedora acceptance**.

## Checked

- 51 Python tests pass with `python -m unittest discover -s tests -v`, including the original 22 tests. The old model-cache test now expects image reuse when only the embedding model changes.
- 10 JavaScript interaction tests pass with `node tests/frontend.test.cjs`: multi-selection, sidebar/workspace drop handling, Uncertain drops, accessible moves, persistent undo requests, accept/reject/skip, failed-save behavior, duplicate suppression, external-drop rejection and cancellation. These use fake event targets, not browser rendering.
- Schema-2 migration adopts valid V1 caches, creates a consistent pre-migration backup, preserves manual history, and rejects a backup symlink escaping into source storage.
- Atomic bulk moves and chained undo survive restarts; revision conflicts and deleted prior groups cannot cause partial or invisible assignments. Accept/reject/skip/undo history persists, reanalysis invalidates stale suggestions, and manual assignments are protected.
- Separate embedding caches, cached CLIP scoring without loading Torch (explicit mock vectors), stale cache export rejection, frame/vector alignment, stage-specific example banks, ambiguous competing groups, hybrid semantic safeguards, and bounded adaptive sampling are tested.
- The reproducible synthetic benchmark runs through the CLI and measures precision, recall, Uncertain coverage, suggestion accuracy, cached scoring time and allocation/RSS measurements. Frozen manifests/results are included under `benchmarks/`; these are handcrafted vectors, not learned-model measurements.
- Real conventional MP4 fixture: ffprobe metadata, five decoded representative frames, embeddings with the offline visual baseline, clustering proposal persistence, full WebM proxy encoding, unchanged-cache reuse.
- Source SHA-256, file size, nanosecond mtime, and directory contents remain unchanged after that workflow.
- Mock decoder exercises BRAW-shaped metadata/frame processing; explicit missing-SDK detection does not fall back to FFmpeg.
- Discovery, case-insensitive BRAW extensions, recursion/symlink exclusions, fingerprints, model-version invalidation, missing-cache rebuild, per-file failures, multi-frame scoring, interleaved groups, and uncertainty behavior.
- SQLite correction persistence and proposal preservation, bulk transaction rollback, empty-only virtual-group deletion, immutable Uncertain group.
- Real in-process ASGI API requests exercise the demo library, mutations, clip details, cached image serving, and host/origin/custom-header restrictions. No network listener is used. A test event-loop timer accommodates the sandbox's suppressed cross-thread wakeup.
- Demo generation produces 22 isolated mock records with five illustrations each; all are explicitly labeled illustrative.
- Python compile checks and JavaScript syntax check pass.
- Diagnostics correctly report available FFmpeg/ffprobe and unavailable BRAW SDK, PyTorch/CLIP dependencies, and model weights.

## Not verified

- Native BRAW bridge **compilation**, runtime ABI, actual decoding, color/exposure, preview, and source integrity with the official SDK. Neither the SDK nor real BRAW footage was available. No claim of working BRAW support is made on the basis of mocks.
- Actual Fedora installation/runtime. Follow README and retain a real BRAW acceptance report plus visual-review notes.
- CLIP inference or GPU acceleration. Implemented but dependencies/weights were unavailable. Only the offline visual baseline was exercised.
- Browser rendering/interaction: Chromium startup failed with `setsockopt: Operation not permitted`; the server could not bind localhost in the Cloud sandbox. The expanded local browser smoke test was attempted and failed at socket creation with `PermissionError: [Errno 1] Operation not permitted`; it has not passed here. API tests are not browser tests.
- Recipe-group accuracy on a professional media library, large-library performance, color-managed/HDR footage, camera rotations/anamorphic footage, long real BRAW proxies, or multi-card recordings.

Dependency/network-enabled tool commands were interrupted without completing; they are not counted as successful installation or validation. Remaining workstation checks are explicitly documented in README.

## Workstation checks for this upgrade

Run the expanded Playwright workflow in `scripts/browser_smoke.py` for actual drag/drop, keyboard moves, preview, group CRUD, recommendation review, screenshots and server-restart persistence. The 51 Python and 10 JavaScript tests do not replace this check. Use the labeled benchmark commands in EVALUATION.md for actual CLIP/baseline/hybrid and uniform/adaptive comparisons. Existing BRAW acceptance remains required; the SDK bridge code was preserved rather than claimed revalidated.
