# Neutral Face Game
A real-time webcam game where the player tries to keep a neutral face while the game tries to make them crack. A facial-expression detector watches each frame; any smile or other expression it catches costs the player points. The project compares a baseline expression model against an improved detector and wires the better one into a playable game.

---

## Table of Contents

1. [How the game works](#how-the-game-works)
2. [Datasets](#datasets)
3. [System architecture](#system-architecture)
4. [Detector output contract](#detector-output-contract)
5. [Repository structure](#repository-structure)
6. [Setup](#setup)
7. [Running the project](#running-the-project)
8. [Team and ownership](#team-and-ownership)
9. [References](#references)

---

## How the game works

First version (v1) scope, to be confirmed at the Nov. 16 kickoff:

| Rule | v1 setting |
|---|---|
| Players | One player at a time |
| Round length | Fixed (e.g., 30 or 60 seconds) |
| Prompts | On-screen text prompts meant to provoke a reaction |
| Penalty | Points lost whenever the detector flags a non-neutral expression |
| Repeated penalties | Cooldown between penalties so one smile isn't counted every frame |
| Missing face | Timer pauses or penalty applies when no usable face is in frame (TBD) |
| Scoreboard | Per-round score, saved between players |

**Fallback plan:** if the full game isn't stable by Nov. 25, v1 narrows to smile detection with text prompts so the model comparison, testing and report can still finish.

---

## Datasets

Neither dataset is committed to this repo. Download them from Kaggle into `data/raw/` (see [Setup](#setup)).

### GENKI-4K
- **Source:** https://www.kaggle.com/datasets/talhasar/genki4k
- **Contents:** ~4,000 face images collected "in the wild" across varied people, lighting and backgrounds.
- **Labels:** binary smile / non-smile, plus head-pose (yaw, pitch, roll).
- **Role in project:** primary data for the smile-vs-neutral task, the core signal for the game.

### FER-2013
- **Source:** https://www.kaggle.com/datasets/msambare/fer2013/data
- **Contents:** ~35,900 grayscale 48×48 face crops, pre-split into `train/` (~28,700) and `test/` (~7,200) folders.
- **Labels:** 7 classes: angry, disgust, fear, happy, sad, surprise, neutral.
- **Role in project:** multi-class expression data, letting the game penalize expressions beyond smiling and giving a larger set for training the improved detector.

### Label mapping
The game only needs a "neutral vs. not neutral" decision, with optional per-expression scores. Proposed mapping:

| Source label | Game label |
|---|---|
| GENKI `smile = 0` | neutral |
| GENKI `smile = 1` | expression (smile) |
| FER `neutral` | neutral |
| FER `happy`, `surprise`, `angry`, `sad`, `fear`, `disgust` | expression |

Caveats to check during data prep:
- GENKI "non-smile" isn't strictly neutral; some non-smile faces carry other expressions.
- FER-2013 has known label noise and a heavy class imbalance (`disgust` is very small).
- Image formats differ (GENKI color, variable size; FER 48×48 grayscale), so preprocessing must normalize both to one input format.

### Splits
Train / validation / test splits are defined once by Person 1 and saved to `data/splits/`. The test split is held out until the models and thresholds are frozen (Nov. 29).

---

## System architecture

```
 Webcam frame
      │
      ▼
 Face detection & crop ──► no face? ──► face_usable = False
      │
      ▼
 Preprocess (grayscale, resize, normalize)
      │
      ▼
 Expression detector (baseline or improved)
      │
      ▼
 Detector output (see contract below)
      │
      ▼
 Game engine: timer · prompts · penalty rules · scoreboard
```

---

## Detector output contract

Every detector, baseline or improved, returns the same structure so the game can swap models without code changes.

```python
{
    "timestamp": 1732060800.123,   # float, seconds (time.time())
    "face_usable": True,           # False if no face, multiple faces, or too blurry/occluded
    "is_neutral": False,           # final decision after thresholding
    "scores": {                    # per-class probabilities, sum to 1
        "neutral": 0.12,
        "smile": 0.81,
        # optional extra classes from FER-2013, e.g. "surprise": 0.04
    },
    "model": "improved_v1"         # which model produced this
}
```

Entry point (owned by Person 2, consumed by Person 3):

```python
from detector import predict
result = predict(frame)  # frame: BGR numpy array from OpenCV
```

---

## Repository structure

```
neutral-face-game/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              # Kaggle downloads (git-ignored)
│   ├── processed/        # preprocessed images (git-ignored)
│   └── splits/           # train/val/test index files (committed)
├── notebooks/
│   ├── 01_data_prep.ipynb          # Person 1
│   ├── 02_baseline.ipynb           # Person 1
│   ├── 03_improved_detector.ipynb  # Person 2
│   └── 04_evaluation.ipynb         # Person 2
├── src/
│   ├── data/             # loading, preprocessing, label mapping
│   ├── models/           # baseline and improved model definitions
│   ├── detector.py       # predict() entry point
│   └── game/             # game loop, timer, scoring, UI
├── models/               # saved weights + config (large files via release or LFS)
├── results/              # metrics, comparison tables, figures, error examples
└── docs/                 # report sections, slides, screenshots
```

---

## Setup

```bash
git clone https://github.com/My1Shadow/neutral-face-game.git
cd neutral-face-game
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Download the data

With the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) and an API token in `~/.kaggle/kaggle.json`:

```bash
kaggle datasets download -d talhasar/genki4k -p data/raw/genki4k --unzip
kaggle datasets download -d msambare/fer2013 -p data/raw/fer2013 --unzip
```

In Colab, upload `kaggle.json` and run the same commands with a `!` prefix.

---

## Running the project

```bash
# 1. Preprocess both datasets and build splits
python -m src.data.prepare

# 2. Train the baseline
python -m src.models.train --model baseline

# 3. Train the improved detector
python -m src.models.train --model improved

# 4. Evaluate both on the held-out test split
python -m src.models.evaluate --models baseline improved

# 5. Play
python -m src.game.main --model improved
```

Commands are placeholders until each module lands; owners update this section as code is merged.

---

## Team and ownership

| Role | Owner | Owns | Report sections (~3 pages each) |
|---|---|---|---|
| Data & baseline | Hyunju Yu| `data/`, `src/data/`, baseline model, combined report assembly | Introduction, dataset, preprocessing, baseline |
| Improved detector & evaluation | Ashley Marx | `src/models/`, `src/detector.py`, evaluation, slide deck assembly | Improved method, model comparison, results, limitations |
| Game & scoring | Erika Gallegos | `src/game/`, README, packaging, system tests | Game design, scoring rules, system testing |

**Git workflow:** work on feature branches (`data/...`, `model/...`, `game/...`), open a pull request into `main`, and get one teammate's review before merging.

---


## References

- Whitehill, J., Littlewort, G., Fasel, I., Bartlett, M., & Movellan, J. (2009). Toward practical smile detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 31(11), 2106–2111. (GENKI-4K)
- Goodfellow, I. J., et al. (2013). Challenges in representation learning: A report on three machine learning contests. *ICONIP 2013*. (FER-2013)
