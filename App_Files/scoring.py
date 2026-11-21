"""Independent, testable game state and event-based scoring.

No GUI or webcam dependencies. Person 2 can change the detector without
changing the rules in this module.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


ROUND_DURATIONS = (20, 30, 45, 60)


def iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Prediction:
    """Common output of all pretrained, simulated, and teammate models.

    smile_score is a model-dependent 0..1 signal and is NOT necessarily a
    calibrated probability of smiling or an emotional state.
    """
    face_detected: bool
    smile_score: float
    timestamp: float
    bbox: Optional[tuple[int, int, int, int]] = None
    source: str = "demo"

    def __post_init__(self):
        if not 0 <= self.smile_score <= 1:
            raise ValueError("smile_score must be between 0 and 1")


@dataclass(frozen=True)
class Event:
    timestamp_utc: str
    player: str
    round_number: int
    game_second: float
    event_type: str
    smile_score: float
    points_delta: int
    detail: str = ""


@dataclass(frozen=True)
class Result:
    player: str
    round_number: int
    points: int
    penalties: int
    active_seconds: float
    duration: int
    completed_at_utc: str


class SmileEventScorer:
    """Debounces detector predictions and counts ONE penalty per smile episode.

    An episode triggers after a smile is sustained for `min_smile_seconds`.
    It cannot trigger again until a neutral reading has been sustained for
    `min_neutral_seconds`. Losing face tracking does not rearm the scorer.
    """

    def __init__(self, threshold: float = 0.5,
                 min_smile_seconds: float = 0.35,
                 min_neutral_seconds: float = 0.55):
        self.threshold = threshold
        self.min_smile_seconds = min_smile_seconds
        self.min_neutral_seconds = min_neutral_seconds
        self.reset()

    def reset(self) -> None:
        self.armed = True
        self._smile_since: Optional[float] = None
        self._neutral_since: Optional[float] = None

    def observe(self, prediction: Prediction, now: float) -> bool:
        """Return True exactly when a new smile penalty should be issued."""
        if not prediction.face_detected:
            self._smile_since = None
            self._neutral_since = None
            return False

        if prediction.smile_score >= self.threshold:
            self._neutral_since = None
            if not self.armed:
                return False
            if self._smile_since is None:
                self._smile_since = now
            if now - self._smile_since >= self.min_smile_seconds:
                self.armed = False
                self._smile_since = None
                return True
            return False

        self._smile_since = None
        if not self.armed:
            if self._neutral_since is None:
                self._neutral_since = now
            if now - self._neutral_since >= self.min_neutral_seconds:
                self.armed = True
                self._neutral_since = None
        return False


@dataclass
class GameEngine:
    duration: int = 30
    penalty_points: int = 10
    face_loss_grace: float = 0.80
    status: str = "ready"  # ready, running, completed
    player: str = ""
    round_number: int = 0
    active_elapsed: float = 0.0
    penalties: int = 0
    manual_paused: bool = False
    tracking_paused: bool = False
    events: list[Event] = field(default_factory=list)
    leaderboard: list[Result] = field(default_factory=list)
    scorer: SmileEventScorer = field(default_factory=SmileEventScorer)
    _last_tick: Optional[float] = None
    _missing_since: Optional[float] = None

    @property
    def points(self) -> int:
        return max(0, 100 - self.penalty_points * self.penalties)

    @property
    def remaining(self) -> float:
        return max(0.0, self.duration - self.active_elapsed)

    @property
    def paused(self) -> bool:
        return self.manual_paused or self.tracking_paused

    def _log(self, kind: str, smile_score: float = 0.0,
             delta: int = 0, detail: str = "") -> None:
        self.events.append(Event(iso_timestamp(), self.player,
                                 self.round_number, round(self.active_elapsed, 2),
                                 kind, round(smile_score, 3), delta, detail))

    def start(self, player: str, now: float, duration: Optional[int] = None) -> None:
        player = player.strip()
        if not player:
            raise ValueError("Please enter a player name")
        if self.status == "running":
            raise RuntimeError("Finish this round before starting another")
        if duration is not None:
            if duration not in ROUND_DURATIONS:
                raise ValueError(f"duration must be one of {ROUND_DURATIONS}")
            self.duration = duration
        self.player = player
        self.round_number += 1
        self.status = "running"
        self.active_elapsed = 0.0
        self.penalties = 0
        self.manual_paused = False
        self.tracking_paused = False
        self._last_tick = now
        self._missing_since = None
        self.scorer.reset()
        self._log("round_started", detail=f"duration={self.duration}s")

    def set_manual_pause(self, pause: bool) -> None:
        if self.status != "running" or self.manual_paused == pause:
            return
        self.manual_paused = pause
        self._log("manual_paused" if pause else "manual_resumed")

    def tick(self, prediction: Prediction, now: float) -> None:
        if self.status != "running":
            return
        if self._last_tick is None:
            self._last_tick = now
        # Avoid counting multi-second browser/server stalls as normal active time.
        delta = max(0.0, min(now - self._last_tick, 0.50))
        self._last_tick = now

        was_tracking_paused = self.tracking_paused
        if prediction.face_detected:
            self._missing_since = None
            self.tracking_paused = False
        else:
            if self._missing_since is None:
                self._missing_since = now
            if now - self._missing_since >= self.face_loss_grace:
                self.tracking_paused = True
            self.scorer.observe(prediction, now)
        if self.tracking_paused != was_tracking_paused:
            self._log("tracking_paused" if self.tracking_paused else "tracking_resumed")

        if self.paused:
            return
        self.active_elapsed = min(float(self.duration), self.active_elapsed + delta)
        if prediction.face_detected and self.scorer.observe(prediction, now):
            self.penalties += 1
            self._log("smile_penalty", prediction.smile_score,
                      -self.penalty_points,
                      "one event; neutral expression required before next")
        if self.active_elapsed >= self.duration:
            self.finish()

    def finish(self) -> None:
        if self.status != "running":
            return
        self.status = "completed"
        self._log("round_completed", detail=f"points={self.points}")
        self.leaderboard.append(Result(
            self.player, self.round_number, self.points, self.penalties,
            round(self.active_elapsed, 2), self.duration, iso_timestamp()))

    def reset_scoreboard(self) -> None:
        """Wipes current leaderboard and event history; preserves round configuration."""
        self.status = "ready"
        self.player = ""
        self.round_number = 0
        self.active_elapsed = 0.0
        self.penalties = 0
        self.manual_paused = False
        self.tracking_paused = False
        self.events.clear()
        self.leaderboard.clear()
        self._last_tick = None
        self._missing_since = None
        self.scorer.reset()

    def export_csv(self, output_dir: str | Path) -> tuple[Path, Path]:
        """Save score/event metadata only. Camera frames are never saved."""
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        scores_path = directory / "scoreboard.csv"
        events_path = directory / "events.csv"
        with scores_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["player", "round", "points", "penalties",
                             "active_seconds", "duration", "completed_at_utc"])
            for row in sorted(self.leaderboard, key=lambda r: (-r.points, r.penalties, r.round_number)):
                writer.writerow([row.player, row.round_number, row.points,
                                 row.penalties, row.active_seconds, row.duration,
                                 row.completed_at_utc])
        with events_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp_utc", "player", "round", "game_second",
                             "event_type", "smile_score", "points_delta", "detail"])
            for row in self.events:
                writer.writerow([row.timestamp_utc, row.player, row.round_number,
                                 row.game_second, row.event_type, row.smile_score,
                                 row.points_delta, row.detail])
        return scores_path, events_path
