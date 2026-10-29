# Keystroke Music: hear a model's accuracy

> **To Claude Code:** complete every phase below **on your own, without asking Atharva anything.** He is not available. Where this file gives a decision rule, apply it and move on. Each checkpoint is a self-check: verify it yourself, log the result in `logs/progress.md`, and continue. Only stop if something is truly impossible, such as no internet. Then write what happened to `SUMMARY.md` and end.

## What this is and why

**Goal: make a classifier's accuracy something a layman can *hear*.**

A model listens to the sound of keys being pressed on a MacBook and guesses which key each click was. Instead of reporting "84% accurate", we turn the guesses into music. A typed phrase becomes a melody, which we play twice: once from the keys that were *actually* pressed and once from the keys the model *heard*. Matching notes mean the model was right. Every wrong note is a key the model misheard. Anyone can hear the difference without knowing what accuracy means.

The model and data come from *"A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards"* (Harrison, Toreini, Mehrnezhad, 2023, arXiv 2308.01074). We use their public recordings. The code is reused from Eshwar Potturi's replication (https://github.com/eshwarpotturi/keystroke-replication). **No new recordings are made.**

### The pipeline
```
paper's recordings (1 file per key, 25 presses each)
  → isolate: cut each file into 25 click clips
  → train: click → key, on presses 1–20 of every key (ordered split)
  → build phrases by stitching held-out clicks (presses 23–25, never trained on)
  → isolate + classify the stitched audio, as an eavesdropper would
  → map true keys and guessed keys to notes
  → one static web page: play clicks / play "what was typed" / play "what the model heard"
```

---

## Phase 0: Setup

**Do:**
- Create the project folder `keystroke-music/` and `git init`. Use this layout:
  ```
  scripts/   data/   models/   logs/   index.html   README.md (this file)   SUMMARY.md
  ```
- Clone sources **outside** the project, into a sibling folder `../sources/`:
  - `git clone --depth 1 https://github.com/eshwarpotturi/keystroke-replication ../sources/keystroke-replication`
  - `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/JBFH-Dev/Keystroke-Datasets ../sources/ksd` (≈570 MB of real WAVs)
- Copy `scripts/isolate.py` and `scripts/train.py` from Eshwar's repo into `scripts/`. Add a credit comment at the top of each.
- Install the dependencies:
  - `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  - `pip install librosa numpy scipy soundfile`
- Add `.gitignore` entries: `data/*.npz`, `__pycache__/`. Never commit `../sources/`.

**Don't:**
- Don't modify anything in `../sources/`.
- Don't set up Colab or a GPU. CPU training takes about 10 minutes.

**✅ Checkpoint 0:** `../sources/ksd/MBPWavs/` contains 36 WAVs, named `0.wav`–`9.wav` and `A.wav`–`Z.wav` (44.1 kHz, stereo, ~20 s each), and `python -c "import torch, librosa"` works.

---

## Phase 1: Isolate the clicks

Use only the **phone recordings** (`MBPWavs/`). Ignore `Zoom/`.

**Do:**
- Adapt `isolate.py` to process only `MBPWavs`. Read the paths case-insensitively: letter files are uppercase, the key labels are lowercase. Key order is `KEYS = "1234567890qwertyuiopasdfghjklzxcvbnm"`.
- Keep the authors' isolator unchanged: `librosa.load(sr=None)` (which mixes down to mono), skip the first second, auto-search the threshold for exactly 25 strokes, use a window of 2400 samples before and 12000 after, and pad to 14400.
- Save `data/phone.npz` with `X, y, pos, sr`, plus `keys=np.array(list(KEYS))`.
- Log the stroke count per key.

**Decision rules:**
- Eshwar's run got 25 strokes on all 36 keys, so this should just work.
- If a key ends up with fewer than 25 strokes, keep what was found. Its last three clips (by position) become its held-out clips. Log it.
- If a key has more than 25, keep the first 25.

**Don't:** don't change the window or clip length. The model's positional embedding needs exactly 14400-sample clips.

**✅ Checkpoint 1:** 36 keys, about 900 clips in total, and the per-key counts are logged.

---

## Phase 2: Train and save the model

**Do:**
- Adapt `train.py`:
  1. Set `nc = len(keys)`, read from the npz.
  2. After training, `torch.save` a dict with `state=best_state`, `mu`, `sd`, `keys`, `sr` to `models/phone.pt`.
  3. Print the test accuracy, a per-key accuracy table, and the 5 most-confused pairs. Save all of it to `logs/train_report.json`.
- Run `THREADS=4 python scripts/train.py phone phone 0 150 ordered`. The ordered split trains on presses 1–20, validates on 21–22 and tests on 23–25.

**Expected:** about 84% test accuracy. Eshwar measured 84.3% on this exact split.

**Decision rules:**
- If the test accuracy is between 60% and 95%, accept it and continue.
- If it's below 60%, something is broken (label order, sample rate, split). Debug once against Eshwar's `train.py` behaviour, then retrain.
- If it's still below 60% after that, continue anyway and state the number honestly on the page and in `SUMMARY.md`.

**Don't:**
- Don't tune the architecture, augmentation or learning rate. Don't run multiple seeds and pick the best.
- **Never** train or validate on presses 23–25.

**✅ Checkpoint 2:** `models/phone.pt` exists, and `logs/train_report.json` has the test accuracy and the confusions.

---

## Phase 3: Build the demo phrases

**Do:**
- Write `scripts/build_demo.py`. It uses only trained keys (0–9, a–z) and treats a space as a pause.
- Use these phrases: `password123`, `hello world`, `straive`, `open the pod bay doors`, `qwerty`, `zoom call 2026`.
- For each phrase:
  1. **Stitch.** For each character, take one of that key's held-out clips (presses 23–25 in `pos`). Cycle through them when a letter repeats. Seed the RNG with 0 so the output is reproducible.
     - Each clip is 14400 samples. The click sits ~2400 samples in, with a tail after it. Place clips so that click onsets are spaced by a random **0.45–0.60 s**. A space adds **1.0 s**.
     - Mix overlapping tails by adding them together, then normalise the whole track to a peak of 0.9.
  2. **Isolate as an eavesdropper would.** Run the same isolator on the full stitched track, searching the threshold until the number of strokes equals the number of non-space characters. Do **not** pass the known clip positions to the model.
  3. **Classify.** Load `models/phone.pt` and apply the same mel features and normalisation as in training. Record the predicted key, confidence and top-3 for each stroke.
  4. **Align.** Match strokes to characters in order: stroke *i* goes with the *i*-th non-space character.
- If the isolator can't hit the exact count for a phrase, widen the gaps (0.6–0.75 s) and retry once. If it still fails, fall back to the known onsets for that phrase only, and set `"isolated_from_audio": false` on it.
- Write `data/demo.json` containing:
  - `test_accuracy`, `n_keys`, `n_test_presses`, from `logs/train_report.json`
  - `phrases`, a list where each item has: `text`, `sr` (16000), `audio` (the track resampled to 16 kHz, int16, base64), `isolated_from_audio`, and `presses` as `[{t, true, pred, conf, top3}]`, where `t` is the onset in seconds within the track.

**Don't:**
- Don't use training or validation clips in the phrases.
- Don't hand-pick clips or phrases to raise accuracy. Use the phrases listed above as they are.
- Don't run the model in the browser. Everything is precomputed here.

**✅ Checkpoint 3:** every phrase is in `demo.json`. Overall phrase accuracy is roughly in line with the test accuracy (within ±15 points is fine with this few presses). Log the per-phrase accuracy.

---

## Phase 4: The page

Build one self-contained `index.html` with plain HTML, CSS and JS, using the Web Audio API. Use no framework and no build step. It loads `data/demo.json`.

### Note mapping (fixed, deterministic; nobody picks notes by ear)
- **Column → pitch.** A key's column is its position from the left within its QWERTY row:
  - `1`=0 … `0`=9
  - `q`=0 … `p`=9
  - `a`=0 … `l`=8
  - `z`=0 … `m`=6

  Column *c* maps to step *c* of this ladder: `C4 D4 E4 G4 A4 C5 D5 E5 G5 A5`. This is a pentatonic scale, so wrong notes sound different rather than ugly.
- **Row → instrument:**
  - number row: `square` wave, low gain
  - top row: `sine` with a fast bell-like decay
  - home row: `triangle`
  - bottom row: `sawtooth` through a low-pass filter at about 1.5 kHz

  Use short attack/decay envelopes so notes don't click. Every key gets a unique (instrument, pitch) pair.
- **Volume:**
  - "What the model heard": `gain = 0.3 + 0.7 × conf`
  - "What was typed": constant gain of 0.8
- **Timing:** both melodies use each press's original onset `t`, so the rhythm matches the clicks.

### Layout
- **Header:** "Can you hear 84% accuracy?" (use the real number). Under it, one plain line: *A model guessed which key was pressed from the sound alone. We turned its guesses into music. Every wrong note is a key it misheard.*
- **One card per phrase**, containing:
  - Three buttons: **▶ Clicks** (the raw audio), **▶ What was typed** (melody from the true keys), **▶ What the model heard** (melody from the predicted keys).
  - The phrase rendered letter by letter, using the model's reading. Wrong letters are red, with the true letter in small text beneath. Spaces show as gaps.
  - During playback, highlight each letter as its note sounds.
  - A small line with the phrase's accuracy, e.g. "10 / 11 keys right".
- **"How it works":** the pipeline in about 4 short lines, plus an honest note:
  - presses were stitched from held-out recordings, not live typing
  - one MacBook, one recording session
  - the paper reports 95% with a larger model and a random split; this is a smaller model on a stricter split
- **Credits:** the paper (with an arXiv link), the dataset repo, and Eshwar's replication.
- Make it readable at 380 px wide, and support light and dark mode with `prefers-color-scheme`.
- No autoplay. Only one sound plays at a time: starting any button stops whatever is playing.

**Don't:**
- Don't add a "type anything" box. Every reading on the page must come from real audio.
- Don't load audio files separately. Everything comes from `demo.json`.
- Don't use external libraries, except Google Fonts if you want a nicer font.

**✅ Checkpoint 4:**
- Serve with `python -m http.server` and fetch the page with curl, confirming it returns 200 and `demo.json` loads.
- Check the HTML for JS syntax errors (e.g. extract the script and run `node --check` if node is available).
- If a headless browser is available, take a screenshot at 380 px and at 1200 px and look at both. If not, review the CSS for mobile layout by reading it.

---

## Phase 5: Publish and summarise

**Do:**
- Commit everything except the gitignored files. The commit message should end with the attribution lines Claude Code normally adds.
- **If** `gh auth status` succeeds:
  1. `gh repo create keystroke-music --public --source . --push`
  2. Enable Pages from `main` / root: `gh api -X POST repos/{owner}/keystroke-music/pages -f "source[branch]=main" -f "source[path]=/"`
  3. Wait for the build to finish and confirm the URL returns 200.
- **If not**, leave the repo committed locally and put the exact two commands Atharva needs to run in `SUMMARY.md`.
- Write `SUMMARY.md` for Atharva (short, plain words):
  1. The live link, or how to publish
  2. The test accuracy and per-phrase accuracy
  3. The 3 most-confused key pairs
  4. Anything that deviated from this plan, and why
  5. A two-line pitch for his manager, e.g. *"Accuracy is abstract to most people, so we made it audible. A keystroke model's guesses play as a melody next to the real one; every wrong note is a mistake. The same trick works for any classifier."*

**Don't:** don't push to Eshwar's repo or any repo other than the new `keystroke-music`.

**✅ Checkpoint 5:** `SUMMARY.md` exists, and the page is either live or ready to publish with two commands.