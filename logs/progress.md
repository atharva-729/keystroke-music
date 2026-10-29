# Progress log

## Checkpoint 0: Setup ✅ (2026-10-29)
- Cloned `../sources/keystroke-replication` and `../sources/ksd` (MBPWavs: 36 WAVs, `0.wav`–`9.wav`, `A.wav`–`Z.wav`, 44.1 kHz stereo, ~21.5 s each).
- Python 3.14.5, torch 2.15.0+cpu, librosa 1.0.0 import fine.
- Deviation: the isolator, mel features and model classes were moved verbatim into `scripts/common.py`, so `isolate.py`, `train.py` and `build_demo.py` share identical code (the original `train.py` runs training on import, so `build_demo.py` could not reuse it otherwise).

## Checkpoint 1: Isolate ✅
- 36 keys, exactly 25 strokes each, 900 clips. Per-key counts and thresholds are in `logs/isolate.log` and `logs/isolate_report.json`.

## Checkpoint 2: Train ✅
- `THREADS=4 python scripts/train.py phone phone 0 150 ordered`, seed 0, single run, no tuning.
- **Test accuracy 86.1%** (93 / 108 held-out presses; Eshwar measured 84.3% on the same split). Best epoch 134 (val 88.9%).
- Most-confused pairs: 6↔8 (2), 7↔8 (2), 1↔q (1), 1↔2 (1), 3↔y (1).
- `models/phone.pt` saved; full report in `logs/train_report.json`, log in `logs/train.log`.

## Checkpoint 3: Demo phrases ✅
- All 6 phrases isolated from the stitched audio with the exact click count at the default 0.45–0.60 s gaps; no retries, no fallbacks.
- password123 10/11 · hello world 8/10 · straive 7/7 · open the pod bay doors 16/18 · qwerty 6/6 · zoom call 2026 10/12.
- **Phrase accuracy 57 / 64 = 89.1%** vs 86.1% test: within ±15 points.
- Deviation: isolation and classification run on the raw-level mix; the 0.9-peak normalisation is applied only to the audio copy used for playback (normalising before classification would shift every feature away from the training normalisation).
- Note: phrases reuse each key's 3 held-out presses, so one misheard press can recur (d→s and o→v each come from one recording).

## Checkpoint 4: Page ✅
- `python -m http.server`: index.html 200, data/demo.json 200 (1.96 MB).
- `node --check` on the extracted script: OK.
- Headless Chrome: screenshots at 1200 px and in a 380 px iframe (light and dark); no horizontal overflow (scrollWidth 360 ≤ 380). Fixed: words breaking mid-word, chart unreadable at 380 px (now drawn at container width), table key ranges wrapping.
- Headless interaction test: 18 play buttons; each mode starts, a second click stops, only one sound at a time; letter and key drill-down dialogs open; no JS errors.
