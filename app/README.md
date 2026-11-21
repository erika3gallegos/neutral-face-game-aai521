# Neutral Face Game 

This is the complete runnable Streamlit game. It currently uses Google's pretrained MediaPipe Face Landmarker, with a drop-in adapter for our team's final model. 

## Run Locally

1. On a Mac, unzip the project; open Terminal and `cd` into this folder.
2. On Apple Silicon with Python 3.13, run:

   ```bash
   python -m pip install -r requirements.txt
   python -m streamlit run streamlit_app.py
   ```

3. Visit http://localhost:8501, start the webcam, grant access, and start a round.
4. Stop Streamlit with Control+C in Terminal.

## Where to Edit

- `streamlit_app.py` - game UI.
- `prompts.py` - the funny prompts.
- `scoring.py` - timer, penalties, and score rules.
- `person2_adapter.py` - replace the pretrained detector later; set `READY = True`.
- `.streamlit/config.toml` - native Streamlit theme colors (hidden in macOS Finder).
- `docs/REPORT_GUIDE.md` - report-writing outline.
- `tests/` - automated tests, kept in their own folder.

- `streamlit_app.py` - Game interface, layout, controls, and display settings.
- `funny_images/` - Add, remove, or replace funny images and GIFs.
- `challenge_media.py` – Image/GIF rotation and timing.
- `scoring.py` – Game timer, smile penalties, and scoring rules.
- `pretrained_model.py` – Pretrained MediaPipe smile detector.
- `person2_adapter.py` – Integrate the custom trained model; set READY = True when ready.
- `report_tools.py` – Scoreboard and CSV exports.
- `config.toml` – Streamlit theme colors.
- `requirements.txt` – Python dependencies for local use and online deployment.

**Do not delete** the other `.py` files: they are imported by the web app.
`requirements.txt` is needed for install and hosting; `packages.txt` is used by
Streamlit Community Cloud for Linux dependencies. `.gitignore` can be kept for GitHub.

## Verify code

```bash
python -m unittest discover -s tests -v
```

## Deploy

Upload the contents of this directory (including the hidden `.streamlit` folder)
to GitHub and deploy `streamlit_app.py` on Streamlit Community Cloud.

A new user/browser session has its own scores; this is not a shared database.
Camera frames are sent to the server for inference but are not deliberately saved.