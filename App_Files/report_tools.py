"""Portable exports; all results are held in the current browser session."""
from __future__ import annotations
import csv
import io
from dataclasses import asdict
from scoring import GameEngine


def scoreboard_rows(game: GameEngine) -> list[dict]:
    """Highest points first; fewer penalties and earlier rounds break ties."""
    ranked = sorted(game.leaderboard, key=lambda r: (-r.points, r.penalties, r.round_number))
    return [
        {"Player": r.player, "Score": r.points, "Penalties": r.penalties,
         "Round": r.round_number, "Duration (s)": r.duration,
         "Active (s)": r.active_seconds}
        for r in ranked
    ]


def _csv_bytes(columns: list[str], rows: list[dict]) -> bytes:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def leaderboard_csv(game: GameEngine) -> bytes:
    cols = ["Player", "Score", "Penalties", "Round", "Duration (s)", "Active (s)"]
    return _csv_bytes(cols, scoreboard_rows(game))


def events_csv(game: GameEngine) -> bytes:
    cols = ["timestamp_utc", "player", "round_number", "game_second",
            "event_type", "smile_score", "points_delta", "detail"]
    return _csv_bytes(cols, [asdict(e) for e in game.events])
