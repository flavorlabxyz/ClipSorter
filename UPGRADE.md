> Historical V1.5 documentation. See [V1_6.md](V1_6.md) and [V1_6_EVALUATION.md](V1_6_EVALUATION.md) for the current release. To reproduce the old benchmark, add `--suite v15` to its `benchmark` commands.

# V1.5: guided sorting

V1.5 preserves the FastAPI application, plain JavaScript frontend, SQLite library, Fedora installation, and official-SDK BRAW bridge. Source paths remain read-only inputs. No cloud AI API or new runtime service is introduced.

## Install without losing your work

1. Stop V1 and keep its application data directory. Back up important corrections using SQLite's backup API or copy the database after shutdown.
2. Extract this release into a separate code directory. Activate your Python environment and run `python -m pip install -e '.[test]'`; add `.[ai]` if you use CLIP. Existing model files under the data directory remain usable.
3. Run `recipe-sorter --data-dir /your/existing/app-data diagnostics`, then `recipe-sorter --data-dir /your/existing/app-data serve`.
4. At database startup, schema 1 migrates to schema 2. A once-only consistent SQLite backup, `library.pre-v1.5.sqlite3`, is stored in app data. Existing human assignments and history survive. Valid V1 embeddings are adopted with their recorded model version.

V1 refuses schema 2. To roll back, stop the server and restore the pre-migration backup into a **separate application data directory** for V1. Human work performed after that backup is not present in it; retain the V1.5 database. Neither upgrade nor rollback requires moving original footage.

## Sorting interactions

- **Sorting workspace** puts Uncertain on the left and recipe groups on the right. Sidebar filters keep Uncertain visible alongside the selected group. Grouped and All footage views, search, status filtering, sampled thumbnails, and previews remain available.
- Select cards with their checkboxes, then drag a selected card to a sidebar group or a workspace group. The entire selection moves, including selected clips outside the current filter. Dragging an unselected card moves only that clip. Drop outlines and a live announcement identify the operation. External file drops are ignored.
- Keyboard/touch alternative: select clips, choose the destination in the bottom bar, and press **Move**. An individual clip's preview dialog also has a group selector and **Save assignment**. Standard Tab/Space/Enter controls work without dragging.
- **New group**, **Rename**, and **Delete empty group** change only SQLite records. Uncertain is permanent. There is no physical sorting or export of originals.
- Every move commits immediately as one transaction. **Undo move** or Ctrl/Cmd+Z outside text inputs reverses the latest move, including a bulk move or accepted recommendation. The undo history survives a restart. Older moves cannot overwrite newer moves on the same clip. Undo refuses to restore a deleted group; use Move to choose a current group.
- Undo restores the previous *manual override*. If there was none, the clip follows its current automatic proposal, which may differ after reanalysis. Undo itself is recorded; redo is not implemented.

## Find matches

1. Move two or more known clips from different recipe stages into a group. One example is allowed, with a limited-coverage explanation. An automatic proposal becomes an example only after you manually confirm/move it.
2. Give competing recipes their own examples too. Without them the system cannot assess whether another group is equally plausible.
3. Click **Find matches** on a group. Check the examples to use, choose local CLIP, the visual baseline, or experimental hybrid, and start the comparison. The selected subset applies to this run; other recipe groups use all their manual examples.
4. Inspect each candidate alongside a matching example frame. The interface shows how many sampled frames have support, the competing group when available, and reasons for ambiguity. Open either image to preview its clip. Similarity values in Comparison details are not probabilities.
5. **Accept into group** creates a normal, undoable manual move. **Reject** stores a negative decision for that group/source fingerprint/backend and suppresses that candidate in later runs for the same combination. **Skip for now** retains its organization and puts it in Skipped; it may appear again in a fresh run. Rejected decisions are not currently resettable in the UI.

Recommendations consider clips with **no manual assignment**, including unreviewed automatic proposals. A manual assignment to any group, including Uncertain, protects the clip from new suggestions. To reconsider such a clip, move it yourself; this release has no “clear all manual decisions” control. Ambiguous candidates remain in their existing group until you act. New unassigned/insufficiently supported automatic clips stay Uncertain.

A run freezes its examples. Accepting one suggestion does not silently turn it into a seed for the remaining cards; start another run to use it. Changing examples, manually moving a candidate, or changing its derived data makes an old suggestion stale. Acceptance rechecks source identity and requires sources online. Cached evidence can still be inspected offline. A changed/offline source never causes a manual move to be overwritten.

The queue displays the latest run for a group. Older runs and all decisions stay in SQLite for future training/evaluation; they are not deleted when another run supersedes the queue. Demo recommendations use baseline descriptors of illustrative images and are not a CLIP/BRAW demonstration.

## Backend and storage changes

`storage/migrations.py` owns the ordered schema-2 migration. Original clips, groups, proposals, corrections, frames, and similarities remain intact.

| Storage | Purpose |
| --- | --- |
| clips.manual_revision / manual_move_id | Optimistic conflict detection and undo ordering |
| clips.sampling_version | Frame cache identity independent of embedding model |
| moves / move_items | Atomic bulk moves, previous manual state, reversal timestamps |
| corrections.move_id / action | Link append-only corrections and undo to the original proposal |
| embeddings | Separate per-clip/backend vectors, ordered frame IDs, fingerprint, model/sampling versions |
| recommendation_runs | Frozen examples, model versions, scoring version and summary |
| suggestions | Ranked candidates, frame evidence, source/derived snapshot and status |
| suggestion_decisions | Accepted, rejected, skipped and undone events with source identity and move reference |

Reanalysis updates automatic proposals and derived data without replacing manual groups. A source fingerprint change invalidates frames/embeddings but preserves human history. Returning from CLIP to the visual baseline no longer discards the other backend's cached embeddings. Model changes regenerate vectors from cached images; sampling changes regenerate both.

New/extended API operations use the existing localhost/origin/custom-header protections:

| Endpoint | Request / behavior |
| --- | --- |
| POST /api/assign | clip_ids, group_id, optional expected_revisions; returns move_id |
| POST /api/moves/{id}/undo | Atomic reversal; conflicts fail without partial changes |
| GET /api/library | Includes last_move and manual revisions |
| POST /api/analyze | backend features/clip/hybrid, force, sampling uniform/adaptive |
| POST /api/recommendations | group_id, backend, optional example_ids; background job |
| GET /api/recommendations/{group_id}?status=pending | Latest run and queue; also skipped/accepted/rejected/undone |
| POST /api/suggestions/{id}/decision | decision accepted/rejected/skipped; acceptance returns move_id |

`analysis/recommendations.py` handles model-independent frame-bank comparison. `backend/recommendations.py` manages jobs, snapshots and decisions. `analysis/embedding_cache.py` manages provenance and lazy model loading. The BRAW decoder/bridge interfaces are unchanged; shared sampling and embeddings still operate on normalized RGB images.

## Run and test

```bash
python -m pip install -e '.[test]'
python -m recipe_sorter --data-dir /tmp/rcs-v15-demo serve --demo
python -m unittest discover -s tests -v
node tests/frontend.test.cjs
node --check recipe_sorter/frontend/app.js
node --check recipe_sorter/frontend/sorting.js
```

Node 18+ is needed only for development tests (`sudo dnf install nodejs`), not normal application use. The ten Node tests exercise the actual shared drag/controller code with fake event targets. They are not rendered-browser tests.

```bash
python -m pip install -e '.[browser]'
python -m playwright install chromium
python scripts/browser_smoke.py
# Or select a compatible installed Chromium:
RCS_BROWSER_EXECUTABLE=/usr/bin/chromium python scripts/browser_smoke.py
```

The Playwright workflow starts an isolated demo server, tests multi-clip sidebar and workspace drops, accessible moves, persisted undo, group management, accept/reject/skip, search/samples/proxy, mobile overflow, and persistence after a server restart. Artifacts stay under `.test-data/browser-smoke-*`. It requires workstation sockets and Chromium support; it has not passed in the restricted Cloud environment.

See [EVALUATION.md](EVALUATION.md) for real labeled evaluation and [README.md](README.md#real-braw-acceptance-on-fedora) for required Fedora/BRAW acceptance. The native bridge and actual local CLIP inference remain unverified here.
