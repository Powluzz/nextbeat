"""Genereert korte, zelfgemaakte (CC0) test-audiofragmenten voor de
Essentia-extractor tests. Geen externe/copyrighted audio nodig.

Draai opnieuw met: python3 tests/fixtures/generate_fixtures.py
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44100
OUT_DIR = Path(__file__).parent


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Schrijft mono float32 samples (range -1..1) weg als 16-bit PCM WAV."""
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def make_click_track(bpm: float, duration_s: float) -> np.ndarray:
    """Kick-drumachtige puls (60 Hz sine + exp. decay) op elke beat, plus een
    ruisburst op elke off-beat ('hihat'), zodat RhythmExtractor2013 iets met
    spectrale flux heeft om op te lokken (een kale klik-trein is te arm)."""
    n_samples = int(duration_s * SAMPLE_RATE)
    audio = np.zeros(n_samples, dtype=np.float64)
    beat_interval = 60.0 / bpm

    t_kick = np.linspace(0, 0.12, int(0.12 * SAMPLE_RATE), endpoint=False)
    kick = np.sin(2 * np.pi * 60 * t_kick) * np.exp(-t_kick * 35)

    t_hat = np.linspace(0, 0.05, int(0.05 * SAMPLE_RATE), endpoint=False)
    rng = np.random.default_rng(42)
    hat = rng.uniform(-1, 1, size=t_hat.shape) * np.exp(-t_hat * 80) * 0.4

    beat = 0
    while True:
        beat_time = beat * beat_interval
        if beat_time >= duration_s:
            break
        start = int(beat_time * SAMPLE_RATE)
        end = min(start + len(kick), n_samples)
        audio[start:end] += kick[: end - start]

        offbeat_time = beat_time + beat_interval / 2
        if offbeat_time < duration_s:
            start_h = int(offbeat_time * SAMPLE_RATE)
            end_h = min(start_h + len(hat), n_samples)
            audio[start_h:end_h] += hat[: end_h - start_h]
        beat += 1

    return audio / max(np.abs(audio).max(), 1e-9) * 0.9


def make_tone(freqs_hz: list[float], duration_s: float) -> np.ndarray:
    """Sustained akkoord (som van sines) — voor key-detectie tests."""
    t = np.linspace(0, duration_s, int(duration_s * SAMPLE_RATE), endpoint=False)
    audio = np.zeros_like(t)
    for f in freqs_hz:
        audio += np.sin(2 * np.pi * f * t)
    # Lichte envelope om klikken aan begin/eind te voorkomen
    fade = int(0.05 * SAMPLE_RATE)
    envelope = np.ones_like(audio)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    audio *= envelope
    return audio / max(np.abs(audio).max(), 1e-9) * 0.8


def main() -> None:
    # 1. Click track @ 128 BPM, 20s — voor BPM-detectie
    click = make_click_track(bpm=128.0, duration_s=20.0)
    _write_wav(OUT_DIR / "click_128bpm.wav", click)

    # 2. C majeur akkoord (C4-E4-G4), 8s — voor key-detectie
    c_major = make_tone([261.63, 329.63, 392.00], duration_s=8.0)
    _write_wav(OUT_DIR / "tone_c_major.wav", c_major)

    # 3. Stilte, 2s — edge case (te stil/leeg voor betekenisvolle analyse)
    silence = np.zeros(int(2.0 * SAMPLE_RATE), dtype=np.float64)
    _write_wav(OUT_DIR / "silence.wav", silence)

    # 4. Corrupt bestand (willekeurige bytes met .wav-extensie) — voor
    # foutafhandelingstest van analyze_audio()
    (OUT_DIR / "corrupt.wav").write_bytes(b"NOT A REAL WAV FILE" * 20)

    print("Fixtures geschreven naar", OUT_DIR)


if __name__ == "__main__":
    main()
