"""Optional local ASR (the upgrade over caption-only fetching).

The paper transcribed podcast audio with Whisper; here that is a pluggable
backend: install `faster-whisper` and audio sources (podcast episodes,
caption-less YouTube videos) are transcribed locally. Without it, the
connectors still work for anything that has captions, and audio-only items
report a clear "no ASR backend" error instead of failing silently.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from . import config


class ASRUnavailable(RuntimeError):
    pass


_transcriber = None
_probed = False


def get_transcriber():
    """Return a transcribe(path)->str callable, or None if no backend."""
    global _transcriber, _probed
    if _probed:
        return _transcriber
    _probed = True
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(config.WHISPER_MODEL, compute_type="int8")

        def transcribe(path: str) -> str:
            segments, _info = model.transcribe(path, vad_filter=True)
            return " ".join(seg.text.strip() for seg in segments)

        _transcriber = transcribe
    except ImportError:
        _transcriber = None
    return _transcriber


def asr_available() -> bool:
    return get_transcriber() is not None


def transcribe_url(audio_url: str) -> str:
    """Download a remote audio file and transcribe it locally."""
    transcriber = get_transcriber()
    if transcriber is None:
        raise ASRUnavailable(
            "No ASR backend installed — `pip install faster-whisper` to "
            "transcribe audio sources without captions."
        )
    import requests

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "audio"
        with requests.get(audio_url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for block in resp.iter_content(1 << 20):
                    f.write(block)
        return transcriber(str(dest))
