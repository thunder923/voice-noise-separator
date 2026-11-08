"""Outdoor mode: synthetic train pass (no speech) + car pass (over speech).
Checks: events are detected on the residual, the no-speech pass is pulled
down much harder with outdoor ON than OFF, speech under the car pass is
mostly preserved, and voice + noise still sums back to the original."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.separator import separate, detect_events

sr = 48000
rng = np.random.default_rng(7)
dur = 14.0
t = np.arange(int(sr * dur)) / sr
rms = lambda x: float(np.sqrt(np.mean(np.square(x))))

# location bed: gentle steady noise
bed = (0.05 * rng.standard_normal(len(t))).astype(np.float32)
mix = bed.copy()

def add_speech(a, b):
    m = (t >= a) & (t < b)
    f0 = 120 + 25 * np.sin(2 * np.pi * 2 * t)
    ph = np.cumsum(2 * np.pi * f0 / sr)
    seg = 0.45 * np.sin(ph) + 0.22 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)
    mix[m] += (seg * (0.7 + 0.3 * np.sin(2 * np.pi * 5 * t)))[m]

add_speech(1.0, 3.0)      # clean dialogue
add_speech(9.5, 11.5)     # dialogue under the car pass

def add_pass(a, b, peak, low_hz=55.0):
    m = (t >= a) & (t < b)
    tt = t[m]
    env = np.sin(np.pi * (tt - a) / (b - a)) ** 1.5
    noise = rng.standard_normal(m.sum()) * 0.8 + 0.9 * np.sin(2 * np.pi * low_hz * tt)
    mix[m] += (peak * env * noise / 1.2).astype(np.float32)

add_pass(5.0, 8.0, 0.55)   # train pass, no speech
add_pass(9.0, 12.5, 0.38)  # car pass, overlaps speech 9.5-11.5

data = np.stack([mix])
no_repair = {k: False for k in ("clicks", "hum", "dropouts", "wind", "ambience")}

off = separate(data.copy(), sr, engine="spectral", strength=0.65, natural=0.2,
               repair_opts={**no_repair})
on = separate(data.copy(), sr, engine="spectral", strength=0.65, natural=0.2,
              repair_opts={**no_repair, "outdoor": True})

for name, r in (("off", off), ("on", on)):
    err = float(np.max(np.abs((r.voice + r.noise) - data)))
    print(f"{name}: sum_err={err:.2e} events={len(r.events)}")
    assert err < 1e-5, "invariant broken"

ev = on.events
print("detected:", [(e["start_sec"], e["end_sec"], e["speech"]) for e in ev])
assert any(e["start_sec"] < 6.5 and e["end_sec"] > 6.5 for e in ev), "train pass not detected"
assert any(e["start_sec"] < 10.5 and e["end_sec"] > 10.5 for e in ev), "car pass not detected"
assert not any(e["start_sec"] >= 0.8 and e["end_sec"] <= 3.2 and abs(e["start_sec"]-1.0) < 0.3 for e in ev), "clean speech mistaken for event"

seg = lambda r, a, b: rms(r.voice[0][int(a*sr):int(b*sr)])
train_off, train_on = seg(off, 5.7, 7.3), seg(on, 5.7, 7.3)
speech_off, speech_on = seg(off, 9.8, 11.2), seg(on, 9.8, 11.2)
clean_on = seg(on, 1.2, 2.8)
print(f"train span voice RMS: off={train_off:.4f} on={train_on:.4f}")
print(f"speech-under-car RMS: off={speech_off:.4f} on={speech_on:.4f} / clean speech={clean_on:.4f}")
assert train_on < train_off * 0.55, "no-speech pass was not pulled down harder"
assert speech_on > clean_on * 0.45, "dialogue under the car pass was gutted"
print("OK: outdoor mode works")
