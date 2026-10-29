# Keystroke Music: summary

## 1. Link
- Code: https://github.com/atharva-729/keystroke-music
- Page (after Pages is switched on): https://atharva-729.github.io/keystroke-music/
- **One step left:** the `gh` CLI isn't installed on this machine, so Pages couldn't be enabled from here. In the repo go to **Settings → Pages → Build and deployment → Deploy from a branch → `main` / `/ (root)` → Save**. The page is live a minute later.
- To run it locally: `python -m http.server`, then open http://localhost:8000.

## 2. Accuracy
- **Test accuracy: 86.1%** (93 of 108 held-out presses, 36 keys, random guessing would get 2.8%). Eshwar got 84.3% on the same split; the paper reports 95% with a larger model and an easier random split.

| Phrase | Keys right |
|---|---|
| password123 | 10 / 11 |
| hello world | 8 / 10 |
| straive | 7 / 7 |
| open the pod bay doors | 16 / 18 |
| qwerty | 6 / 6 |
| zoom call 2026 | 10 / 12 |
| **All phrases** | **57 / 64 (89%)** |

The isolator found every click from the audio on its own in all six phrases.

## 3. Most-confused key pairs
1. `6` ↔ `8` (2 times)
2. `7` ↔ `8` (2 times)
3. `1` ↔ `q`, `1` ↔ `2` and `3` ↔ `y` (once each)

12 of the 15 mistakes were a neighbouring key.

## 4. Where I deviated from the plan
- **Repo:** pushed to your existing `atharva-729/keystroke-music` (as you asked), not a new `gh repo create`.
- **Normalisation:** each stitched track is classified at its raw recording level. The 0.9-peak normalisation is applied only to the copy you hear. Normalising first would rescale the audio away from what the model was trained on, which isn't what a real microphone would hand an eavesdropper.
- **Shared code:** the isolator and model classes were copied unchanged into `scripts/common.py`, so isolation, training and the demo all use identical code.
- **Page layout:** it follows the README's layout inside your usual results-page structure: problem and approach first, a playable phrase on the first screen, then the data, the phrases, a clickable per-key keyboard (each key opens its 3 test presses), a confidence chart, and below a divider ("Everything below is the how"), the method, honest notes, note mapping, confused pairs, how to reproduce, and credits.
- **Worth knowing:** phrases reuse each key's 3 held-out presses, so one misheard recording can show up in several phrases (`d`→`s` appears 3 times but is one press). The page says so.

## 5. Two-line pitch
*Accuracy is abstract to most people, so we made it audible. A keystroke model's guesses play as a melody next to the real one: every wrong note is a mistake, and the same trick works for any classifier.*
