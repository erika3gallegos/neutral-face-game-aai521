"""Select local Try Not to Laugh images/GIFs for the current round.

Drop .jpg, .jpeg, .png, .gif or .webp files in the funny_images folder.
Images change every six *active* seconds (pauses do not advance them).
"""
from pathlib import Path

MEDIA_DIR = Path(__file__).resolve().parent / "funny_images"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
SECONDS_PER_IMAGE = 6


def media_files() -> list[Path]:
    """Return available media files in predictable alphabetical order."""
    if not MEDIA_DIR.is_dir():
        return []
    return sorted(
        (file for file in MEDIA_DIR.iterdir()
         if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda file: file.name.lower(),
    )


def current_media(active_elapsed: float) -> tuple[int, Path | None, int]:
    """Return (1-based image number, file path, total images)."""
    files = media_files()
    if not files:
        return 0, None, 0
    index = max(0, int(active_elapsed // SECONDS_PER_IMAGE)) % len(files)
    return index + 1, files[index], len(files)
