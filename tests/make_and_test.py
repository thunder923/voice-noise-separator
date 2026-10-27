"""Synthetic verification: speech-like bursts + pink-ish noise + room hum.
Checks: stems are written, voice+noise == original, noise is reduced during
speech pauses in VOICE, and speech energy is mostly preserved in VOICE."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.separator import separate_file, save_wav, load_audio

sr = 48000
rng = np.random.default_rng(42)
dur = 6.0
t = np.arange(int(sr * dur)) / sr

# noise: filtered white noise (wind / room tone) + 100Hz hum
white = rng.standard_normal(len(t)).astype(np.float32)
b = np.cumsum(white) * 0.02
b -= np.linspace(0, b[-1], len(b))
noise_src = (0.25 * white * 0.3 + b / (np.max(np.abs(b)) + 1e-9) * 0.35
             + 0.05 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)

# speech-like: harmonic bursts with formant-ish envelope, 3 phrases with pauses
voice_src = np.zeros_like(t)
for start, end in [(0.5, 2.0), (2.6, 4.0), (4.6, 5.6)]:
    m = (t >= start) & (t < end)
    f0 = 120 + 30 * np.sin(2 * np.pi * 2 * t)
    ph = np.cumsum(2 * np.pi * f0 / sr)
    seg = (0.5 * np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph))
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 5 * t)
    voice_src[m] = (seg * env)[m]

mix = np.stack([voice_src + noise_src])  # mono [1, n]
inp = Path(__file__).parent / "test_input.wav"
outd = Path(__file__).parent / "out"
save_wav(inp, mix, sr)

info = separate_file(inp, outd, engine="spectral", strength=0.85)
print(info)
assert info["sum_max_error"] < 1e-4, "stems do not sum back to original!"

voice, _ = load_audio(info["voice_path"])
noise, _ = load_audio(info["noise_path"])
v, n = voice[0], noise[0]

pause = (t >= 2.1) & (t < 2.5)   # no speech here
speech = (t >= 0.8) & (t < 1.8)  # speech here
rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
print(f"pause : original noise RMS {rms(noise_src[pause]):.4f} -> VOICE {rms(v[pause]):.4f} / NOISE stem {rms(n[pause]):.4f}")
print(f"speech: original voice RMS {rms(voice_src[speech]):.4f} -> VOICE {rms(v[speech]):.4f} / NOISE stem {rms(n[speech]):.4f}")
assert rms(v[pause]) < rms(noise_src[pause]) * 0.55, "voice stem still noisy in pauses"
assert rms(v[speech]) > rms(voice_src[speech]) * 0.55, "voice stem lost the dialogue"
print("OK: separation works, stems sum to original")
