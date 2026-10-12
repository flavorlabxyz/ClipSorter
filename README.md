# Recipe Clip Sorter V1.5

A local, source-safe workbench for finding recipe threads in mixed raw footage. Discover clips, compare multiple frames, review suggested groups, preview clips, and save your own organization. Python + FastAPI, SQLite, plain browser UI. Designed for one Fedora Linux workstation; no Docker, cloud AI service, or paid API.

**V1.5 does not identify recipes.** It suggests which clips look related. An explicit **Uncertain** group holds clips without a sufficiently clear match. Your manual decisions always take precedence.

**BRAW implementation status:** a native bridge to the official Blackmagic RAW Linux SDK is included as source. The SDK and real BRAW footage were unavailable during development. **The bridge has not been compiled against the SDK or validated on real BRAW footage on Fedora.** Its presence, mock tests, and demo thumbnails do not establish working BRAW integration. Follow the acceptance procedure below before relying on it.

## What works in V1.5

- Recursive or top-level discovery of `.braw`, `.mov`, `.mp4`, `.m4v`, `.mkv`, `.avi`, `.mts`, `.mxf`, and `.webm`.
- Metadata normalization, source fingerprints, incremental frame/embedding caching, individual failure states, background progress, and persistent SQLite organization.
- Five sparse representative samples, near 10%, 30%, 50%, 70%, and 90%; nearby retries for dark, blurred, or unreadable frames. An optional adaptive experiment adds up to four samples around visual changes.
- A replaceable local embedding interface: optional learned **CLIP** image embeddings, plus an immediately usable **offline visual baseline** using color, layout, and texture. The baseline is not a learned AI model and is labeled accordingly.
- A sorting workspace with Uncertain beside recipe groups, single/multiple clip dragging onto sidebar or workspace groups, sample strips, search, filters, accessible individual/bulk moves, and persistent undo.
- Example-based recommendations using manually assigned clips: compare against several recipe stages, inspect supporting frame pairs, accept/reject/skip, and preserve decisions for later evaluation.
- Independent CLIP/baseline caches, experimental hybrid scoring, and reproducible labeled benchmarks.
- Disposable, silent WebM review proxies. Conventional footage uses FFmpeg; BRAW RGB frames originate only from the official SDK bridge, then FFmpeg encodes them.
- Append-only automatic proposals and correction history, separate from the current effective assignment.
- Environment diagnostics, isolated illustrative demo library, automated tests, and a real-BRAW acceptance command.

Not included: recipe naming, transcription, ingredient/action recognition, temporal food-state reasoning, editing timelines, exporting/copying media, physical sorting, cloud analysis, collaborative access, or training on your corrections.

## Upgrading from V1

Stop the old server, keep your existing application data directory, and install this version with `python -m pip install -e '.[test]'`. Launch with the same `--data-dir`. Schema 2 is applied automatically and a SQLite backup is created at `library.pre-v1.5.sqlite3` inside that data directory. Manual groups, corrections, and proposals are preserved. Do not open the upgraded database with V1. See [UPGRADE.md](UPGRADE.md) for interactions, API/storage changes, and undo behavior; [EVALUATION.md](EVALUATION.md) for the CLIP audit, benchmark commands, measured synthetic results, and limitations.

## Fedora installation

Use a current supported Fedora release, Python 3.11 or newer, and a modern Firefox/Chromium browser. CPU-only operation is supported; CLIP is slower without a GPU. Use a Linux x86-64 workstation and SDK package that matches its architecture for BRAW; verify the vendor's requirements for your SDK release.

```bash
sudo dnf install python3 python3-pip gcc-c++ ffmpeg-free
cd /path/to/recipe-clip-sorter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
recipe-sorter diagnostics
recipe-sorter serve
```

Open **http://127.0.0.1:8765**. Stop with Ctrl+C. Do not bind it to a public network; the app intentionally accepts localhost hosts/origins only. Use one process per library (no Uvicorn `--workers` or autoreload).

`ffmpeg-free` capabilities vary by Fedora release. Both `ffmpeg` and `ffprobe` must be on PATH, and FFmpeg needs the `libvpx-vp9` encoder for proxies. Check your actual build:

```bash
ffmpeg -version
ffprobe -version
ffmpeg -encoders | grep libvpx
```

If FFmpeg cannot decode one of your source codecs, use a Fedora-compatible build providing that decoder. [RPM Fusion's multimedia instructions](https://rpmfusion.org/Howto/Multimedia) explain Fedora's optional full multimedia packages. Installing third-party repositories is a workstation decision; the app does not do it automatically. There is **no dependency on FFmpeg's BRAW support**, even if a particular build advertises it.

### Local learned image embeddings

Install the optional AI dependencies and explicitly download CLIP once:

```bash
source .venv/bin/activate
python -m pip install -e '.[ai]'
recipe-sorter prepare-model
```

Then choose **Local CLIP model** in the Analyze dialog, or:

```bash
recipe-sorter analyze --backend clip
```

The model is `openai/clip-vit-base-patch32`, downloaded through Hugging Face by `prepare-model`. See its [model card](https://huggingface.co/openai/clip-vit-base-patch32). Allow disk space for PyTorch and model weights. For a smaller CPU-only PyTorch installation, follow [PyTorch's official Linux installer](https://pytorch.org/get-started/locally/) before installing this project's AI extra. The app uses CUDA when available, otherwise CPU; BRAW decoding in V1 always uses the SDK CPU pipeline.

Analysis uses `local_files_only=True` and does not download models or send images anywhere. If CLIP dependencies or weights are missing, analysis gives a setup error; it **never silently substitutes the baseline**. The default baseline works without model downloads and is useful for exercising real-media processing, but needs careful human review.

## Using the application

1. Click **Add footage**, enter an absolute local folder path, and choose whether to include subfolders. A browser cannot safely provide an arbitrary server-side folder picker; the path field is intentional.
2. Scan. Discovery records path, name, extension, size, nanosecond filesystem mtime, decoder, fingerprint, and available recording/media metadata. Missing recording dates remain unknown; filesystem mtime is not presented as recording time.
3. Click **Analyze clips**, select the visual backend, and start analysis. The worker processes one clip at a time and reports per-file problems. Continue reviewing other clips while it runs.
4. Review suggested recipe threads. Click the small bars on a card to switch sampled frames, or open the clip for all samples, metadata, group controls, and proposal evidence.
5. Generate a proxy on demand. The complete clip is available as a reduced-resolution **silent** preview. BRAW playback targets 6 fps; conventional playback targets 12 fps. Long clips can take time. Playback appears when the job finishes.
6. In **Sorting workspace**, review Uncertain beside the recipe groups. Select cards, then drag one selected card onto a sidebar group or workspace group. Dragging an unselected card moves only that card. Alternatively use the bottom **Move** bar or the clip dialog. **Undo move** / Ctrl+Z restores the preceding manual state, including after restarting.
7. Manually place a few examples from different stages into a recipe group. Click **Find matches**, choose the examples and backend, and review the ranked candidates. Accept moves a clip into the group; reject records a negative example; skip keeps it available for later review. No suggestion moves footage until you accept it.
8. Create or rename groups, move clips to Uncertain, and delete only empty virtual groups.
9. **Every mutation is saved immediately in SQLite.** There is no unsaved batch that requires a global Save button. Reopen with the same data directory to retain results and corrections.

The sidebar filters to a group; the sorting workspace also retains the Uncertain column. “All footage” gives a flat view. Status filters distinguish unprocessed footage, successful analysis, manual review, and failures. Unavailable BRAW clips remain visible and can be assigned virtually even before the SDK is installed.

If source paths become unavailable, metadata and thumbnails remain available but new processing/playback requires reconnecting the sources. Rescan after external source changes. V1 does not auto-match moved/renamed originals to old database records.

### Command-line workflow

Place `--data-dir` **before** the subcommand. Stop a server using the same library before CLI scans, analysis, or acceptance checks.

```bash
recipe-sorter --data-dir ~/.local/share/recipe-sorter-session scan /mnt/footage/shoot-01
recipe-sorter --data-dir ~/.local/share/recipe-sorter-session analyze --backend clip
recipe-sorter --data-dir ~/.local/share/recipe-sorter-session serve
```

Use `scan --no-recursive`, `analyze --force`, or `serve --port 8770` as needed. The equivalent uninstalled invocation from the project directory is `python -m recipe_sorter ...`.

## Blackmagic RAW installation and architecture

Obtain the **official Blackmagic RAW SDK for Linux** from [Blackmagic Design's developer downloads](https://www.blackmagicdesign.com/developer/products/braw/sdk-and-software). Review its license and installation requirements. This repository includes no SDK headers, libraries, installer, or redistributable runtime. Use the headers, dispatch source, and libraries from the **same official SDK release**; do not substitute third-party modified headers.

The included bridge targets the conventional clip/frame CPU interfaces documented in the [official Blackmagic RAW SDK manual](https://documents.blackmagicdesign.com/DeveloperManuals/BlackmagicRAW-SDK.pdf). The build script detects the newer audio callback declaration where present. SDK API changes may require bridge adjustments; an uncompiled bridge must not be considered verified.

Extract the SDK outside your footage directory, preserve its Linux library layout, then:

```bash
source .venv/bin/activate
python scripts/build_braw.py --sdk "$HOME/SDKs/Blackmagic RAW SDK"
export RCS_BRAW_BRIDGE="$PWD/native/recipe-braw"
export BLACKMAGIC_RAW_LIBRARY_DIR="$HOME/SDKs/Blackmagic RAW SDK/Linux/Libraries"
# Use the actual directory containing libBlackmagicRawAPI.so in your SDK package.
recipe-sorter diagnostics
recipe-sorter serve
```

If your package uses an architecture subdirectory, point `BLACKMAGIC_RAW_LIBRARY_DIR` there. Keep required companion libraries together. If the loader reports a missing dependency, use `ldd` on the vendor library to identify it and install the appropriate Fedora package; do not disable SELinux or replace system libraries. Persistent environment variables may be placed in your own shell or desktop launcher configuration.

The build uses `g++`, the SDK's `BlackmagicRawAPI.h`/Linux headers, and its `BlackmagicRawAPIDispatch.cpp`. It does not install anything globally. Run the bridge directly for a narrowly scoped diagnostic:

```bash
"$RCS_BRAW_BRIDGE" diagnostic
"$RCS_BRAW_BRIDGE" probe /mnt/footage/clip.braw
```

`MediaDecoder` exposes normalized metadata, RGB frames, and generated preview material. `FFmpegDecoder` handles conventional formats. `BRAWDecoder` invokes the isolated native bridge:

- Opens a clip with the official SDK and reads dimensions, frame rate, frame count/duration, camera type, timecode, and scalar clip metadata.
- Reads frames at requested positions using SDK asynchronous read/decode/process jobs.
- Chooses SDK half/quarter/eighth resolution where sufficient, requests RGBA8 CPU output, and converts it to a common RGB representation. Full resolution is used only when smaller scales cannot supply the requested sample size.
- Clones processing attributes in memory and requests Rec.709 gamma/gamut. If the SDK rejects an override, it retains the SDK clip default and writes a diagnostic to stderr. Color/exposure must be checked on your footage.
- Supplies sparse frames to the shared sampling/embedding pipeline. For a proxy, one bridge process opens the source once and streams 6-fps RGB frames to FFmpeg for WebM encoding. FFmpeg does not open the `.braw` source.
- Never invokes trim, metadata setters on a clip, sidecar writes, or source output operations. Existing sidecars may be read by the SDK, not rewritten.

The bridge protocol consists of `diagnostic`/`probe` JSON and `frame`/`stream` binary PPM output on stdout; errors go to stderr. Codecs and SDK-specific structures do not escape into analysis, storage, or the UI. Missing SDK/bridge errors are recorded as `braw_unavailable`; non-BRAW workflows continue normally.

### Real BRAW acceptance on Fedora

**This procedure is required before declaring BRAW support validated.** Use a small real clip first, then clips from each camera/recording mode you use. Keep sources on read-only storage if possible. Choose an application data directory outside the source tree.

```bash
# Prepare learned embeddings in the acceptance library if using --backend clip.
recipe-sorter --data-dir ~/.local/share/rcs-braw-check prepare-model
recipe-sorter --data-dir ~/.local/share/rcs-braw-check diagnostics
recipe-sorter --data-dir ~/.local/share/rcs-braw-check braw-acceptance /mnt/footage/test.braw --backend clip
recipe-sorter --data-dir ~/.local/share/rcs-braw-check serve
```

`--backend features` can test the decoding path without CLIP installation, but does not establish learned-model integration. Acceptance writes a dated JSON report under `acceptance/` inside the chosen data directory. It checks discovery, decoder selection, clip opening, duration, frame rate, resolution, useful camera/timecode metadata, multiple sampled positions, persisted embeddings, similarity calculation, a persisted clustering proposal, proxy creation, and source SHA-256/size/mtime equality. It also checks directory entries and existing sidecars for changes.

Finish these **human checks**:

1. In the UI, find the real clip and compare all sampled thumbnails against a trusted BRAW player. Confirm reasonable color, exposure, orientation, and recognizable content.
2. Play and seek the generated proxy. Confirm its sequence and duration correspond to the source. BRAW previews intentionally omit audio.
3. Add several real clips from an interleaved filming session, analyze, and inspect similarity proposals and Uncertain behavior. A lone clip is deliberately Uncertain; a successful single-clip acceptance run does not demonstrate recipe-group accuracy.
4. Move the clip, rename its group, restart the app, and verify your correction persists. Re-run the acceptance command and inspect `source_unchanged`, `directory_entries_unchanged`, and `sidecars_unchanged`.
5. Retain the report, Fedora release, SDK version, camera/recording settings, and your visual-review notes as your acceptance record.

`automated_checks_passed` covers only automated checks. Reports always retain a human-review-pending list and do not automatically set `braw_fully_validated` to true. A report on another distribution is not Fedora acceptance. Hashing a large BRAW file twice takes time; there is no partial-hash shortcut in this test.

## Source safety

Original media is an **input only**. V1 contains no source move, rename, delete, metadata-update, copy/export, or physical reorganization feature. All group changes update database rows. The app never deliberately writes next to a source file.

- Generated images, proxies, embeddings, SQLite files, logs, model downloads, subprocess temporaries, and acceptance reports are rooted in application data.
- Scan rejects overlapping source/data trees, including resolved symlink aliases. Discovery does not follow symlink directories or files; decoding rejects a source that has become a symlink.
- Output paths are validated against the canonical data root. Cache path traversal and symlink escape are rejected. No arbitrary filesystem path is served to the browser; frames and proxies are resolved by database IDs.
- Decoder processes receive argument arrays, not shell commands, and run with the application data directory as their working directory. Their temporary directories are also inside application data.
- Fingerprints are checked before/after processing; results are rejected if a source changes while it is being analyzed or proxied.
- Tests hash fixture source contents and compare size/mtime and directory inventory after real decoding, sampling, analysis, proxies, and virtual organization. Test media is made read-only before analysis. Fixtures are synthesized into temporary source directories during setup, never altered by analysis.

These are application-level guarantees, not a kernel sandbox for vendor libraries. A read-only mounted source volume is the strongest additional protection against any dependency defect or filesystem race. Reading files may update filesystem **access time** according to your mount options; the app does not change source content, embedded metadata, mtime, or names. Avoid changing source files concurrently, and use trusted local media/dependencies. This is a single-user local application, not a hardened service for hostile uploads.

## Data, cache, and persistence

Default: `$XDG_DATA_HOME/recipe-clip-sorter`, or `~/.local/share/recipe-clip-sorter`. Override with `RCS_DATA_DIR` or `--data-dir`. Choose an app-owned directory, **never a footage folder**. Directory creation occurs at startup; the app cannot know an unselected source root in advance.

```text
library.sqlite3          media, metadata, frames/embeddings, virtual groups,
                        proposal runs, pair evidence, correction/move/recommendation history
library.pre-v1.5.sqlite3  once-only pre-migration backup
benchmarks/              labeled manifests and evaluation reports
frames/                 disposable sampled JPEGs
previews/               disposable WebM files
models/                 CLIP weights and local model caches
logs/                   rotating application logs
tmp/                    subprocess temporary storage
acceptance/             real-media safety/acceptance reports
demo-library/           isolated demo database and generated illustrations
```

Back up the SQLite database if corrections are important. Stop the app before copying it, or use SQLite's backup API; an active WAL database must not be backed up by copying only the main file. Do not delete the database to clear thumbnails: it holds your human work. There is no automatic cache eviction in V1. Old image generations and partial proxies can accumulate; stop the app before manually clearing the disposable `frames/` or `previews/` directories. Missing sampled images are rebuilt on analysis. Missing proxies can be generated again from the clip review.

Fingerprints combine canonical path, size, and nanosecond mtime. Unchanged clips with matching sampling/model versions reuse frames and embeddings. A model/version change reuses existing sampled JPEGs and regenerates only stale embeddings. CLIP and visual features coexist in separate cache records. Changing the sampling strategy or **Rebuild cached analysis** regenerates frames and their embeddings. Pairwise grouping is recalculated on Analyze and receives its own run version. Fingerprints are efficient, not cryptographic content identities; changes preserving both size and mtime require forced analysis. The acceptance workflow uses full SHA-256.

Schema management uses SQLite `PRAGMA user_version`; V1.5 installs schema 2, enables foreign keys and WAL, backs up an existing schema-1 database before migration, and refuses an unknown newer schema. Separate tables retain per-backend embeddings, atomic moves/undo, frozen recommendation runs, and append-only review decisions. A process lock prevents concurrent CLI/server analysis in a library. Interrupted clips are marked failed at the next server/CLI startup and can be retried.

Deleted virtual groups are tombstoned so historical proposal/correction references remain valid. A correction stores the original current proposal, previous manual assignment, chosen group, confidence/evidence, proposal ID, source fingerprint, and timestamp. Re-analysis creates new proposals without replacing correction history or manual assignments.

## How automatic grouping works

V1.5 retains this V1 baseline for comparison. **Find matches** adds a separate, human-guided example-bank method described in [EVALUATION.md](EVALUATION.md); it does not require every recipe clip to resemble every other clip. Human examples guide comparisons immediately, but no model is fine-tuned.


Each usable frame gets its own normalized embedding; V1 keeps all sampled embeddings, not just a first-frame vector. Pair scoring combines symmetric nearest-frame coverage and clip-centroid similarity. Recording-time proximity, matching camera identity, and matching dimensions provide small, bounded bonuses. Filesystem chronology never determines grouping.

Complete-link visual consistency prevents long chains of weak neighbors from joining a thread. A clip needs enough visual support and a margin over the second candidate. Ambiguous assignments and singletons remain Uncertain. Evidence retains best/second candidate, visual support, score margin, threshold, reason, and the top five pair explanations per clip per analysis run. Scores are **similarities, not calibrated probabilities**.

This is a conservative appearance-based starting point. Similar bowls/countertops can still produce false matches, and different stages of the same recipe can split apart. Thresholds have not been calibrated on your media. Review results before trusting them. The embedding and evidence-provider abstractions support later ingredient/action, speech, temporal state, camera, timeline, or correction-derived evidence without changing media decoding. V1.5 uses manually assigned clips as examples for recommendations and records decisions; it does not train model weights.

Analysis is sequential and sparse; CLIP batches each clip's sampled frames. Pair comparison is quadratic and V1 limits a library to **1,500 real clips** for analysis. Use separate data directories for larger sessions. V1 has no persistent queue or cancellation UI. A corrupted clip is recorded and skipped; a failed model setup stops the job with instructions. Proxies are generated on demand, capped by a two-hour subprocess timeout.

## Demo and development

```bash
recipe-sorter serve --demo
# Without installing the package, using already available dependencies:
python -m recipe_sorter --data-dir /tmp/rcs-demo serve --demo
```

Demo mode automatically appends `demo-library` to the data root and prevents real scanning/analysis. It provides 22 synthetic clip records, illustrative sampled images, three example named threads, uncertain examples, and illustrative confidence values. A visible banner labels it. BRAW filenames are mock labels and have never been decoded. A generated demo preview is a five-second still illustration, not source video. Changes made to demo groups persist in the demo library. Assign demo clips to groups manually before using Find matches. Demo recommendations compute actual visual-baseline descriptors from the illustrations; CLIP and hybrid requests in demo are explicitly refused.

The UI needs no frontend build, npm packages, CDN assets, or external fonts. The default server binds only to `127.0.0.1`. Cloud environments that prohibit localhost sockets cannot launch its web server; the demo files and in-process API tests still work there. No public hosted site is created.

## Tests and diagnostics

```bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
recipe-sorter diagnostics
node tests/frontend.test.cjs                 # 10 interaction tests; Node 18+
node --check recipe_sorter/frontend/app.js
# Optional real browser workflow on a workstation:
python -m pip install -e '.[browser]'
python -m playwright install chromium
python scripts/browser_smoke.py
```

Tests cover discovery/extension handling/BRAW selection, normalized metadata, database persistence, transactions, virtual groups, correction history, uncertainty and interleaved visual threads, multiframe scoring, fingerprints/cache invalidation, unavailable SDK handling, individual failures, dark frames, API origin/host protections, and source safety. A real FFmpeg integration test generates a small conventional clip, decodes samples, creates a proxy, verifies cache reuse, and compares source hashes/size/mtime. It skips explicitly if FFmpeg/ffprobe are absent. Other BRAW tests use mocks only.

The API tests use in-process ASGI transport. They do not require a listening server or exercise browser rendering. Run `python scripts/browser_smoke.py` on a workstation with Playwright and its Chromium installed for UI interaction, persistence, and screenshots (see script docstring). The Cloud execution sandbox blocked browser startup and listening sockets during this build; browser rendering was not validated there.

Diagnostics report platform/distribution, Python, FFmpeg and ffprobe versions, SDK bridge/runtime availability, CLIP dependency/weight availability, baseline provider, CPU thread count, CUDA status where PyTorch is installed, and data/database paths. The active analysis provider/version is recorded per clip and in proposal runs. Missing SDK functionality has its own status; the rest of the application remains usable.

## Structure

```text
recipe_sorter/
  media/discovery.py          read-only discovery
  media/decoders/             shared interface, FFmpeg, official-SDK adapter
  analysis/                  sampling, embeddings, evidence, clustering
  storage/                   safe cache paths, SQLite schema and history
  backend/                   API, background service, diagnostics
  frontend/                  local review UI
  demo.py                    isolated illustrative library
  acceptance.py              opt-in real BRAW validation
  cli.py                     launch/setup/diagnostic commands
native/bridge.cpp            only SDK-specific native code
scripts/build_braw.py        compile using separately installed official SDK
scripts/browser_smoke.py     local browser workflow check
tests/                      automated unit, API, and FFmpeg integration tests
```

See [VALIDATION.md](VALIDATION.md) for what was actually checked in the development environment and what remains unverified.
