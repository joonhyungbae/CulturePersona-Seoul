# annotation — human-audit tool

The web tool used for the registered human audit of the narratives (paper appendix on the human audit; the
analysis is `scripts/anno_analysis.py`). Two Korean raters used it; the interface text is Korean and is kept
exactly as the raters saw it.

## Protocol the tool enforces

1. **Stage 1, blind.** The rater reads a narrative with the anchors hidden and records a first judgement.
2. **Stage 2.** Only after the stage-1 answer is stored does the server release the key for that item; the
   anchors and the original narrative are then shown and the rater scores the item. Only the first stage-1
   answer is kept.
3. **Independent scores** are locked once both raters have finished; a rater never sees the other rater's
   scores before that.
4. **Adjudication** opens after both have finished and covers only the items on which the raters disagreed
   about a contradiction.

The public item file (`public/data/items.json`) holds the masked narratives in plain text; anchors and original
narratives are sealed per item (AES-256-GCM) and unsealed in the browser with the released key.

## Parts

| Part | Where | What |
| --- | --- | --- |
| Interface | `src/` (Vite + React + TypeScript), served as a static site | Rating, adjudication and progress views |
| Backend | `apps_script/Code.template.js` (Google Apps Script with a spreadsheet) | Access codes, stage-1 record and key release, score storage, locking, adjudication, request log |
| Local server | `server/annoPlugin.ts` (Vite plugin) | The same protocol, for local testing |
| Data build | `scripts/build_public.mjs` | Builds the sealed `public/data/items.json` and the backend script from the private item file |
| Tests | `scripts/protocol_test.py`, `scripts/apps_script_emulator.mjs` | Protocol checks against the local server or an emulator of the backend |

The local server and the Apps Script backend implement the same rules.

## Not in the repository

The answer key (stratum, trap and checker labels per item), the item keys and unsealed originals
(`data_private/`), the generated backend script with the keys embedded (`apps_script/Code.gs`), the raters'
access codes and the rating records. The sample itself is drawn by `scripts/anno_build_sample.py` and reduced by
`scripts/anno_reduce_sample.py` in the main repository.

## Running locally

```bash
cd annotation
npm install
npm run typecheck
./run_local.sh                  # local server on 127.0.0.1:5180; ratings go to ../outputs/anno/ (git-ignored)
```

`npm run build:data` needs `data_private/`, which is not distributed. `dev_ux.sh` starts a development instance
with a sandbox store and placeholder codes (`DEVR1`, `DEVR2`, `DEVADJ`, `DEVOWNER`).
