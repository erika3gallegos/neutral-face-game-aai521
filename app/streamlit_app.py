"""Neutral Face Game | Free-hostable Streamlit + browser WebRTC prototype.

Person 3 owns gameplay, rules, and system testing. Person 2 owns only the
replaceable expression detector implemented in person2_adapter.py.
"""
from __future__ import annotations

import os
import time
import base64
from pathlib import Path

from PIL import Image
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from detectors import build_detector
from challenge_media import current_media
from report_tools import events_csv, leaderboard_csv, scoreboard_rows
from scoring import GameEngine
from webcam_bridge import LatestPrediction, WebcamProcessor, simulated_prediction


st.set_page_config(
    page_title="Neutral Face Game", page_icon="😐", layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* University of San Diego inspired palette, from the official brand guide. */
:root {
    --usd-founders: #003B70;
    --usd-immaculata: #0074C8;
    --usd-torero: #75BEE9;
    --usd-pale: #F0F7FD;
}
.stApp {background: linear-gradient(180deg, #FFFFFF 0%, #F7FBFF 55%, #FFFFFF 100%);}
.block-container {padding-top: 1.65rem; max-width: 1280px;}
.block-container::before {
    content: ""; display: block; height: 5px; margin: 0 0 1.35rem;
    border-radius: 6px;
    background: linear-gradient(90deg, #003B70 0%, #0074C8 68%, #75BEE9 100%);
}
.hint {color: #315571; font-size: 1.02rem; line-height: 1.55;}
/* Readable body typography: captions, form labels, status text and metrics. */
/* These selectors target Streamlit components, not the custom USD header/footer. */
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p {
    color: #315571 !important;
    font-size: 1.02rem !important;
    line-height: 1.58 !important;
    font-weight: 500 !important;
}
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p {
    color: #173F61 !important;
    font-size: 1.03rem !important;
    line-height: 1.45 !important;
    font-weight: 650 !important;
}
.stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stMetricLabel"] p,
.stApp [data-testid="stMetricLabel"] div {
    color: #315571 !important;
    font-size: 0.98rem !important;
    font-weight: 750 !important;
    letter-spacing: 0.02em;
}
.stApp [data-testid="stAlert"] p {
    color: #173F61 !important;
    font-size: 1.03rem !important;
    line-height: 1.5 !important;
}
.stApp [data-testid="stExpanderDetails"] p {
    color: #244761;
    font-size: 1.02rem;
    line-height: 1.6;
}
/* Game-show inspired header using the same USD blues as the footer. */
.usd-game-hero {
    position: relative; isolation: isolate; overflow: hidden;
    box-sizing: border-box; border-radius: 20px;
    margin-bottom: 1.6rem;
    padding: clamp(1.5rem, 3.5vw, 2.7rem) clamp(1.5rem, 4vw, 3.1rem);
    color: #FFFFFF;
    background: linear-gradient(112deg, #003B70 0%, #004D8B 62%, #0074C8 100%);
    border: 1px solid rgba(0,59,112,0.16);
    box-shadow: 0 12px 34px rgba(0, 59, 112, 0.15);
}
.usd-game-hero::before {
    content: ""; position: absolute; z-index: -1;
    height: 360px; width: 360px; border-radius: 50%;
    border: 65px solid rgba(117, 190, 233, 0.11);
    right: -152px; top: -170px;
}
.usd-game-hero::after {
    content: ""; position: absolute; z-index: -1;
    width: 42%; height: 220%; right: 8%; top: -45%;
    transform: rotate(27deg);
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.035), transparent);
}
.usd-game-hero-inner {
    display: flex; justify-content: space-between; align-items: center;
    gap: 1.5rem; position: relative; z-index: 1;
}
.usd-game-hero-copy {max-width: 760px; min-width: 0;}
.usd-game-eyebrow {
    display: inline-flex; align-items: center; gap: 0.55rem;
    color: #C6E9FC; font-size: 1.1rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: 0.145em;
    margin-bottom: 0.95rem;
}
.usd-game-eyebrow::before {
    content: ""; display: block; width: 8px; height: 8px;
    border-radius: 50%; background: #75BEE9;
    box-shadow: 0 0 0 4px rgba(117,190,233,0.15);
}
.usd-game-title {
    font-family: inherit; font-size: clamp(2.5rem, 5.2vw, 4.4rem);
    font-weight: 850; letter-spacing: -0.055em;
    line-height: 1.03; margin: 0; color: #FFFFFF;
}
.usd-game-title-accent {color: #9DD9FA;}
.usd-game-lead {
    color: #E2F2FD; font-size: clamp(0.97rem, 1.45vw, 1.14rem);
    line-height: 1.5; max-width: 570px; margin: 0.9rem 0 1.25rem;
}
.usd-game-chips {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
}
.usd-game-chip {
    display: inline-block;
    padding: 0.45rem 0.85rem;
    border: 1px solid rgba(198,233,252,0.25);
    border-radius: 8px;
    background: rgba(255,255,255,0.05);
    color: #D4EAF9;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.015em;
    line-height: 1.4;
}
.usd-game-visual {
    position: relative; flex: 0 0 185px; height: 195px;
    display: flex; align-items: center; justify-content: center;
}
.usd-game-face-ring {
    display: flex; justify-content: center; align-items: center;
    width: 152px; height: 152px; border-radius: 50%;
    background: linear-gradient(145deg, #FFFFFF 0%, #DDEFFB 100%);
    border: 7px solid rgba(255,255,255,0.28);
    box-shadow: 0 14px 30px rgba(0,27,59,0.23);
    transform: rotate(-7deg);
}
.usd-game-face {font-size: 5.2rem; line-height: 1; transform: rotate(7deg);}
.usd-game-sticker {
    position: absolute; right: -4px; bottom: 7px;
    padding: 0.62rem 0.7rem; border-radius: 8px;
    background: #75BEE9; color: #003B70;
    box-shadow: 0 6px 14px rgba(0,27,59,0.17);
    font-size: 0.65rem; font-weight: 850; letter-spacing: 0.07em;
    transform: rotate(7deg); white-space: nowrap;
}
@media (max-width: 760px) {
    .usd-game-hero {border-radius: 15px;}
    .usd-game-visual {flex-basis: 110px; height: 140px;}
    .usd-game-face-ring {width: 102px; height: 102px; border-width: 5px;}
    .usd-game-face {font-size: 3.7rem;}
    .usd-game-sticker {font-size: 0.5rem; bottom: 0; right: -4px;}
}
@media (max-width: 530px) {
    .usd-game-hero-inner {align-items: flex-start;}
    .usd-game-visual {display: none;}
    .usd-game-title {font-size: clamp(2.45rem, 10vw, 3.4rem);}
    .usd-game-eyebrow {font-size: 0.64rem;}
    .usd-game-chip {font-size: 0.85rem;}
}
.demo-face {text-align: center; background: linear-gradient(145deg, #EAF5FC, #F7FBFF);
            border: 1px solid #C9E1F3; border-radius: 18px; padding: 2.35rem 0;}
.demo-face .emoji {font-size: 7.3rem; line-height: 1.2;}
.demo-face .tag {color: #315D80; letter-spacing: 0.11em;
                font-size: 0.8rem; font-weight: 700; margin-top: 0.35rem;}
.prompt-card {padding: 1.15rem 1.3rem; background: #EAF5FC;
              border: 1px solid #BBDDF3; border-radius: 14px; margin: 0.6rem 0 1rem;}
/* Compact challenge heading makes room for the whole image and scoreboard. */
.prompt-card--compact {padding: 0.65rem 1rem; margin: 0.45rem 0 0.65rem;}
.prompt-card .label {font-size: 0.93rem; letter-spacing: 0.08em;
                     color: #005B9E; font-weight: 750;}
.prompt-card .sentence {font-size: clamp(1.05rem, 2vw, 1.3rem); line-height: 1.4;
                        color: #003B70; font-weight: 650; margin-top: 0.45rem;}
.score-summary {font-size: 0.9rem; color: #315D80; padding-bottom: 0.4rem;}
/* Native Streamlit primary buttons use the matching theme in .streamlit/config.toml. */
footer {visibility: hidden;}

/* Footer typography mirrors the larger, bolder game header. */
.usd-project-footer {
    position: relative; overflow: hidden; box-sizing: border-box;
    margin-top: 2.8rem; border-radius: 16px;
    background: #003B70; color: #FFFFFF;
    border-top: 5px solid #75BEE9;
    box-shadow: 0 10px 28px rgba(0, 59, 112, 0.13);
    padding: clamp(1.6rem, 3vw, 2.5rem);
    font-family: inherit;
}
.usd-project-footer::after {
    content: ""; position: absolute; width: 280px; height: 280px; border-radius: 50%;
    right: -120px; top: -155px; border: 44px solid rgba(117,190,233,0.10);
    pointer-events: none;
}
.usd-footer-grid {
    display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr);
    align-items: center; gap: clamp(1.8rem, 3.5vw, 3.5rem);
    position: relative; z-index: 1;
}
.usd-footer-left {display: flex; gap: 1.2rem; align-items: center; min-width: 0;}
.usd-footer-left > div {min-width: 0;}
.usd-footer-logo {max-height: 75px; max-width: 180px; object-fit: contain; flex-shrink: 0;}
.usd-footer-eyebrow {
    display: flex; align-items: center; gap: 0.55rem;
    font-size: 1.05rem; text-transform: uppercase;
    letter-spacing: 0.09em; font-weight: 800; line-height: 1.4;
    color: #C6E9FC; margin-bottom: 0.8rem;
}
.usd-footer-eyebrow::before {
    content: ""; display: block; flex-shrink: 0;
    width: 8px; height: 8px; border-radius: 50%;
    background: #75BEE9;
    box-shadow: 0 0 0 4px rgba(117,190,233,0.15);
}
.usd-footer-university {
    font-family: inherit; font-size: clamp(1.85rem, 2.6vw, 2.4rem);
    line-height: 1.13; letter-spacing: -0.035em;
    font-weight: 800; color: #FFFFFF;
}
.usd-footer-course {
    font-size: 1.08rem; line-height: 1.5;
    color: #E2F2FD; margin-top: 0.75rem;
}
.usd-footer-team {min-width: 0;}
.usd-footer-authors {
    font-size: 1.22rem; color: #FFFFFF;
    font-weight: 700; line-height: 1.5;
}
.usd-footer-meta {
    font-size: 1.05rem; line-height: 1.5;
    color: #E2F2FD; margin-top: 0.7rem;
}
@media (max-width: 940px) {
    .usd-footer-grid {grid-template-columns: 1fr; gap: 1.7rem;}
    .usd-footer-team {border-top: 1px solid rgba(198,233,252,0.25); padding-top: 1.5rem;}
}
@media (max-width: 650px) {
    .usd-project-footer {padding: 1.5rem 1.25rem;}
    .usd-footer-left {align-items: flex-start; flex-direction: column; gap: 0.9rem;}
    .usd-footer-logo {max-height: 60px;}
    .usd-footer-eyebrow {font-size: 0.94rem; letter-spacing: 0.06em;}
    .usd-footer-university {font-size: 1.9rem;}
    .usd-footer-authors {font-size: 1.15rem;}
    .usd-footer-course, .usd-footer-meta {font-size: 1rem;}
}
</style>
""", unsafe_allow_html=True)


def initialize() -> None:
    if "engine" not in st.session_state:
        st.session_state.engine = GameEngine(duration=30)
    if "latest_predictions" not in st.session_state:
        st.session_state.latest_predictions = LatestPrediction()
    if "detector_mode" not in st.session_state:
        st.session_state.detector_mode = "MediaPipe pretrained (webcam)"
    if "locked_detector_mode" not in st.session_state:
        st.session_state.locked_detector_mode = st.session_state.detector_mode
    if "player" not in st.session_state:
        st.session_state.player = "Player 1"
    if "round_duration" not in st.session_state:
        st.session_state.round_duration = 30
    if "smile_threshold" not in st.session_state:
        st.session_state.smile_threshold = 0.45
    if "sim_face" not in st.session_state:
        st.session_state.sim_face = True
    if "sim_smile" not in st.session_state:
        st.session_state.sim_smile = False


def mode_changed() -> None:
    game = st.session_state.engine
    if game.status == "running":
        st.session_state.detector_mode = st.session_state.locked_detector_mode
        st.session_state.mode_notice = "Finish the round before switching detectors."
    else:
        st.session_state.locked_detector_mode = st.session_state.detector_mode
        st.session_state.latest_predictions.clear()
        st.session_state.pop("processor_mode", None)


def rtc_config() -> dict | None:
    """If optional Cloudflare secrets exist, let streamlit-webrtc discover TURN.

    Otherwise use public STUN (not guaranteed to work from every cloud/network).
    """
    try:
        secret_id = st.secrets.get("CLOUDFLARE_TURN_KEY_ID")
        secret_token = st.secrets.get("CLOUDFLARE_TURN_KEY_API_TOKEN")
    except Exception:  # No secrets file is expected for the free STUN-only demo.
        secret_id = secret_token = None
    if secret_id and secret_token:
        os.environ["CLOUDFLARE_TURN_KEY_ID"] = str(secret_id)
        os.environ["CLOUDFLARE_TURN_KEY_API_TOKEN"] = str(secret_token)
    if (os.getenv("CLOUDFLARE_TURN_KEY_ID") and
            os.getenv("CLOUDFLARE_TURN_KEY_API_TOKEN")):
        return None  # Library fetches short-lived TURN credentials automatically.
    return {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


@st.cache_data(show_spinner=False)
def fitted_media_width(media_file: str, max_height: int = 250,
                       max_width: int = 520) -> int:
    """Size the whole image/GIF to fit, preserving its original aspect ratio.

    Streamlit's st.image width is specified in pixels. Reading only image
    metadata avoids cropping portraits or tall GIFs at a fixed display width.
    """
    try:
        with Image.open(media_file) as image:
            image_width, image_height = image.size
        if image_width <= 0 or image_height <= 0:
            return 300
        scale = min(max_width / image_width, max_height / image_height, 1.0)
        return max(1, round(image_width * scale))
    except (OSError, ValueError):
        return 300


initialize()
# To use Person 2's model later, update this single setting after completing
# person2_adapter.py (including READY = True). The dropdown stays hidden.
ACTIVE_DETECTOR = "MediaPipe pretrained (webcam)"
st.session_state.detector_mode = ACTIVE_DETECTOR
game: GameEngine = st.session_state.engine
# The detector produces a model-specific score. Calibrate a demo threshold here.
game.scorer.threshold = float(st.session_state.smile_threshold)

st.html("""
<section class="usd-game-hero" aria-label="Neutral Face Game introduction">
  <div class="usd-game-hero-inner">
    <div class="usd-game-hero-copy">
      <div class="usd-game-eyebrow">AAI-521 · Final Project · Team 7</div>
      <h1 class="usd-game-title">Neutral Face <span class="usd-game-title-accent">Game</span></h1>
      <p class="usd-game-lead">Think you can keep a straight face? Beat the clock without cracking a smile.</p>
      <div class="usd-game-chips" aria-label="Game rules">
        <span class="usd-game-chip">START WITH 100 POINTS</span>
        <span class="usd-game-chip">EVERY SMILE COSTS 10 POINTS</span>
      </div>
    </div>
    <div class="usd-game-visual" aria-hidden="true">
      <div class="usd-game-face-ring"><span class="usd-game-face">😐</span></div>
      <div class="usd-game-sticker">NO SMILING!</div>
    </div>
  </div>
</section>
""")

st.markdown("### Set Up Round")
st.caption("Get ready to test your neutral face!")
config1, config2 = st.columns([1.2, 0.8])
with config1:
    st.text_input("Player Name", key="player", max_chars=40)
with config2:
    st.selectbox("Round Duration", options=[20, 30, 45, 60],
                 key="round_duration", format_func=lambda x: f"{x} seconds")

st.slider(
    "Smile Sensitivity Threshold (lower = more sensitive)",
    min_value=0.20, max_value=0.85, step=0.05,
    key="smile_threshold", disabled=game.status == "running",
)
game.scorer.threshold = float(st.session_state.smile_threshold)

if st.session_state.pop("mode_notice", None):
    st.info("Finish the current round before switching detectors.")

is_simulation = st.session_state.detector_mode == "Simulation (no webcam)"
left, right = st.columns([1.1, 0.92], gap="large")

with left:
    st.markdown("### Camera")
    if is_simulation:
        face = "😐" if st.session_state.sim_face and not st.session_state.sim_smile else "😄"
        if not st.session_state.sim_face:
            face = "🚫"
        st.markdown(
            '<div class="demo-face"><div class="emoji">' + face +
            '</div><div class="tag">SIMULATED MODEL OUTPUT · NO CAMERA</div></div>',
            unsafe_allow_html=True,
        )
        a, b = st.columns(2)
        with a:
            st.toggle("Face visible", key="sim_face")
        with b:
            st.toggle("Smile detected", key="sim_smile", disabled=not st.session_state.sim_face)
        st.caption("Use these two switches to test the scoring and pause rules. "
                   "They simulate model predictions; they do not measure facial expressions.")
        webcam_context = None
    else:
        detector_mode = ("MEDIAPIPE PRETRAINED"
                         if st.session_state.detector_mode.startswith("MediaPipe")
                         else "PERSON 2 MODEL")
        if st.session_state.get("processor_mode") != detector_mode:
            try:
                detector = build_detector(detector_mode)
                st.session_state.processor = WebcamProcessor(
                    detector, st.session_state.latest_predictions)
                st.session_state.processor_mode = detector_mode
                st.session_state.latest_predictions.clear()
            except Exception as exc:
                st.error(f"Unable to initialize detector: {exc}")
                st.stop()
        st.caption(
            "To enable your webcam, click 'START CAMERA' below "
            "and allow access when prompted. Once your camera is active, "
            "you're ready to play!")
        webcam_context = webrtc_streamer(
            key=f"neutral-face-camera-{detector_mode.lower().replace(' ', '-')}",
            video_frame_callback=st.session_state.processor.on_frame,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration=rtc_config(),
            async_processing=True,
            translations={
                "start": "START CAMERA",
                "stop": "STOP CAMERA",
            },
        )
        st.caption("Having trouble connecting your webcam? "
                   "Check your browser's camera permissions.")
        
    with st.expander("How scoring works", expanded=False):
        st.markdown("""
        Players start each round with **100 points**.
        
        - **Smile penalty:** Smiling for at least 0.35 seconds results in a 10-point penalty.
        - **Repeated smiles:** A continuous smile counts as one penalty. After returning to a neutral expression for at least 0.55 seconds, another smile can trigger a new penalty.
        - **Face tracking:** If your face is not detected for 0.80 seconds, the timer pauses until tracking resumes.
        - **Try Not to Laugh:** Funny images and GIFs change every six seconds of active gameplay.
        
        **Goal:** Finish the round with as many points as possible!
        """)

with right:
    st.markdown("### Live Round")

    @st.fragment(run_every=0.20)
    def live_round() -> None:
        g: GameEngine = st.session_state.engine
        now = time.monotonic()
        camera_running = is_simulation or bool(webcam_context and webcam_context.state.playing)
        if is_simulation:
            prediction = simulated_prediction(
                face_visible=st.session_state.sim_face,
                smile=st.session_state.sim_smile,
                now=now,
            )
            detector_error = None
        else:
            prediction, detector_error = st.session_state.latest_predictions.read(now)
            if not camera_running:
                prediction = simulated_prediction(face_visible=False, smile=False, now=now)
        if g.status == "running":
            g.tick(prediction, now)

        score_col, time_col, pen_col = st.columns(3)
        score_col.metric("POINTS", g.points)
        time_col.metric("TIME LEFT", f"{g.remaining:04.1f}s")
        pen_col.metric("PENALTIES", g.penalties)

        if g.status == "ready":
            status = None  # No redundant instruction before the round starts.
        elif g.status == "completed":
            status = f"Round complete: {g.player} finished with {g.points} points."
        elif g.manual_paused:
            status = "Manually paused. Resume to continue."
        elif g.tracking_paused:
            status = "Paused: face not detected. Return to view to continue."
        elif not prediction.face_detected:
            status = "Finding face… countdown pauses after 0.8 seconds without tracking."
        elif prediction.smile_score >= g.scorer.threshold:
            status = "Smile signal detected! Keep a neutral expression."
        else:
            status = "Neutral expression — keep it up!"
        if status:
            st.info(status)
        if detector_error:
            st.error(detector_error)

        # Rotate funny images and GIFs every six seconds of active play.
        # The sequence stays consistent across players for comparable rounds.
        if g.status == "running":
            media_number, media_path, media_total = current_media(g.active_elapsed)
        else:
            media_number, media_path, media_total = 0, None, 0

        st.markdown(
            '<div class="prompt-card prompt-card--compact">'
            '<div class="label">TRY NOT TO LAUGH</div></div>',
            unsafe_allow_html=True,
        )
        if media_path:
            # Show the FULL image/GIF, never cropping its top or bottom.
            # Limiting display height keeps the scoreboard in view.
            st.image(str(media_path),
                     width=fitted_media_width(str(media_path)))
        elif g.status == "running":
            st.warning("No images found. Add JPG, PNG, or GIF files to funny_images/.")

        c1, c2, c3 = st.columns([1.1, 1, 1])
        with c1:
            if st.button("▶ Start Round", type="primary", use_container_width=True,
                         disabled=(g.status == "running" or not camera_running),
                         help="Begin a new round for the name entered above"):
                try:
                    g.start(st.session_state.player, now=time.monotonic(),
                            duration=st.session_state.round_duration)
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))
                else:
                    st.rerun(scope="fragment")
        with c2:
            if st.button("⏸ Pause Round" if not g.manual_paused else "▶ Resume Round",
                         use_container_width=True, disabled=g.status != "running"):
                g.set_manual_pause(not g.manual_paused)
                st.rerun(scope="fragment")
        with c3:
            if st.button("⏹ End Round", use_container_width=True,
                         disabled=g.status != "running"):
                g.finish()
                st.rerun(scope="fragment")

        if not camera_running and not is_simulation:
            st.caption("Connect the webcam before starting a round.")

        st.markdown("### Scoreboard")
        rows = scoreboard_rows(g)
        if rows:
            # Allow additional players to scroll instead of extending the page.
            score_height = min(155, max(80, 38 + len(rows) * 35))
            st.dataframe(rows, use_container_width=True, hide_index=True,
                         height=score_height)
        else:
            st.caption("Completed rounds will appear here. The highest score wins.")

        d1, d2 = st.columns(2)
        with d1:
            st.download_button("↓ Download Scoreboard", data=leaderboard_csv(g),
                               file_name="neutral_face_scoreboard.csv",
                               mime="text/csv", use_container_width=True,
                               disabled=not rows)
        with d2:
            st.download_button("↓ Download Event Log", data=events_csv(g),
                               file_name="neutral_face_events.csv", mime="text/csv",
                               use_container_width=True, disabled=not g.events)
        if st.button("Reset All Players and Scores", disabled=g.status == "running"):
            g.reset_scoreboard()
            st.rerun(scope="fragment")

    live_round()

# Optional official USD logo. Only add an approved image at assets/usd_logo.png.
logo_path = Path(__file__).resolve().parent / "assets" / "usd_logo.png"
logo_html = ""
if logo_path.is_file():
    logo_data = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    logo_html = (
        f'<img class="usd-footer-logo" '
        f'src="data:image/png;base64,{logo_data}" '
        f'alt="University of San Diego approved logo">'
    )

st.html(
    f"""
    <div class="usd-project-footer" role="contentinfo">
      <div class="usd-footer-grid">
        <div class="usd-footer-left">
          {logo_html}
          <div>
            <div class="usd-footer-eyebrow">Fall 2026</div>
            <div class="usd-footer-university">University of San Diego</div>
            <div class="usd-footer-course">AAI-521 · Applied Computer Vision for AI</div>
          </div>
        </div>
        <div class="usd-footer-team">
          <div class="usd-footer-eyebrow">Team 7 · Project Authors</div>
          <div class="usd-footer-authors">Ashley Marx · Erika Gallegos · Hyunju Yu</div>
          <div class="usd-footer-meta">Neutral Face Game · Course Final Project</div>
        </div>
      </div>
    </div>
    """,
)
