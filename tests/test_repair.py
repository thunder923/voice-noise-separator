"""Verify repair stages on synthetic wireless/hum/dropout damage."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.repair import declick_channel, detect_hum_base, remove_hum_channel, repair_dropouts_channel

sr = 48000
rng = np.random.default_rng(7)
t = np.arange(sr * 4) / sr
# clean speech-like signal
f0 = 130 + 20 * np.sin(2 * np.pi * 2 * t)
ph = np.cumsum(2 * np.pi * f0 / sr)
clean = (0.4 * np.sin(ph) + 0.2 * np.sin(2 * ph)).astype(np.float32)
clean *= (0.7 + 0.3 * np.sin(2 * np.pi * 3 * t)).astype(np.float32)

# --- clicks: 30 random impulses (wireless RF hits)
x = clean.copy()
pos = rng.choice(len(x), 30, replace=False)
x[pos] += rng.choice([-1, 1], 30) * 1.5
y, n = declick_channel(x, sr)
peak_before = np.max(np.abs(x - clean))
peak_after = np.max(np.abs(y - clean))
print(f"clicks: repaired {n} samples, peak error {peak_before:.2f} -> {peak_after:.2f}")
assert peak_after < peak_before * 0.25, "declick did not suppress impulses"
assert np.sqrt(np.mean((y - clean) ** 2)) < 0.05, "declick damaged the signal"

# --- hum/buzz: 50 Hz + harmonics (Japan east = 50 Hz)
# dialogue with pauses (hum detection listens to the pauses, like real takes)
gate = np.zeros_like(t)
for s, e in [(0.3, 1.6), (2.2, 3.5)]:
    gate[(t >= s) & (t < e)] = 1.0
from scipy.ndimage import uniform_filter1d
dialogue = clean * uniform_filter1d(gate, 200)
buzz = sum(0.08 / k * np.sin(2 * np.pi * 50 * k * t) for k in range(1, 12)).astype(np.float32)
x2 = dialogue + buzz
clean = dialogue  # hum error is measured against the paused dialogue
base = detect_hum_base(x2, sr)
print("hum detected base:", base)
assert base == 50.0
y2 = remove_hum_channel(x2, sr, base)
band = lambda s: float(np.mean(s[np.abs(t % 1.0) < 2] ** 2))
err_before = np.sqrt(np.mean((x2 - clean) ** 2)); err_after = np.sqrt(np.mean((y2 - clean) ** 2))
print(f"hum: RMS error {err_before:.4f} -> {err_after:.4f}")
assert err_after < err_before * 0.4, "hum removal insufficient"

# --- dropout: tear a 60 ms hole inside a phrase (0.8 s; a hole inside a
# pause is just a longer pause and correctly needs no repair)
x3 = clean.copy()
a, b = int(0.8 * sr), int(0.86 * sr)
x3[a:b] = 0.0
y3, n3 = repair_dropouts_channel(x3, sr)
gap_rms = float(np.sqrt(np.mean(y3[a:b] ** 2)))
print(f"dropout: repaired {n3} samples, gap RMS after = {gap_rms:.4f} (was 0)")
assert n3 > 0 and gap_rms > 0.02, "dropout not bridged"

print("OK: all repair stages work")
