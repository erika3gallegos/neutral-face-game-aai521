# Neutral Face Game | Clean Project Folder

University of San Diego - Fall 2026 - AAI-521, Team 7

This is the **complete runnable Streamlit game** with USD-blue theme, webcam, Google's
pretrained MediaPipe Face Landmarker, smile scoring, leaderboard, and a drop-in
adapter for Person 2. It is not a patch: you do not need other ZIPs.

## Run locally

1. On a Mac, unzip the project; open Terminal and `cd` into this folder.
2. On Apple Silicon with Python 3.13, run:

   ```bash
   python -m pip install -r requirements.txt
   python -m streamlit run streamlit_app.py
   ```

3. Visit http://localhost:8501, start the webcam, grant access, and start a round.
4. Stop Streamlit with Control+C in Terminal.

Python 3.13 on Intel macOS may need a separate older Python environment for MediaPipe.
On first use, the app downloads Google's Face Landmarker model to `models/`
(approximately 4 MB); that folder appears automatically. The model is not in this ZIP.

## Where to edit

- `streamlit_app.py` - game UI, USD theme CSS, footer, and `ACTIVE_DETECTOR` choice.
- `prompts.py` - the funny prompts.
- `scoring.py` - timer, penalties, and score rules.
- `person2_adapter.py` - replace the pretrained detector later; set `READY = True`.
- `.streamlit/config.toml` - native Streamlit theme colors (hidden in macOS Finder).
- `docs/REPORT_GUIDE.md` - report-writing outline.
- `tests/` - automated tests, kept in their own folder.

**Do not delete** the other `.py` files: they are imported by the web app.
`requirements.txt` is needed for install and hosting; `packages.txt` is used by
Streamlit Community Cloud for Linux dependencies. `.gitignore` can be kept for GitHub.
The app does not need the optional `download_model.py`, setup notes, or example logos
from earlier ZIPs. If you add an *approved* USD logo, use `assets/usd_logo.png`.

## Verify code

```bash
python -m unittest discover -s tests -v
```

## Deploy

Upload the contents of this directory (including the hidden `.streamlit` folder)
to GitHub and deploy `streamlit_app.py` on Streamlit Community Cloud.
WebRTC may require TURN on restrictive networks. Never commit real keys.

A new user/browser session has its own scores; this is not a shared database.
Camera frames are sent to the server for inference but are not deliberately saved.
