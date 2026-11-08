"""Post-normalize settings: custom target, voice basis, peak ceiling.
Checks the voice stem actually lands on the target LUFS, the stems still
sum back (to the gained original), and the ceiling caps the gain."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.separator import separate_file, load_audio
from core.loudness import measure_lufs

src = Path(__file__).parent / "out" / "loud_in.wav"
out = Path(__file__).parent / "out" / "loud_res"
out.mkdir(parents=True, exist_ok=True)

sr = 48000
rng = np.random.default_rng(3)
t = np.arange(sr * 6) / sr
voice = (0.30 * np.sin(2*np.pi*140*t) + 0.15 * np.sin(2*np.pi*280*t)).astype(np.float32)
voice[int(sr*4.5):] *= 0.15
mix = voice + (0.04 * rng.standard_normal(len(t))).astype(np.float32)
from core.separator import save_wav
save_wav(src, np.stack([mix]), sr)

no_repair = {k: False for k in ("clicks", "hum", "dropouts", "wind", "ambience")}

# 1) custom target on the VOICE basis
info = separate_file(src, out, engine="spectral", repair_opts=no_repair,
                     loudness="custom", loudness_target=-18.0,
                     loudness_basis="voice", loudness_ceiling_db=-1.0)
v, _ = load_audio(info["voice_path"])
got = measure_lufs(v, sr)
print(f"voice basis: target=-18.0 measured_after={got:.2f} info={info['loudness']}")
assert abs(got - (-18.0)) < 0.4, "voice stem did not land on target"
assert info["sum_max_error"] < 1e-4

# 2) preset still works (mix basis, default ceiling)
info2 = separate_file(src, out, engine="spectral", repair_opts=no_repair,
                      loudness="streaming")
print("preset:", info2["loudness"])
assert info2["loudness"]["target_lufs"] == -14.0 and info2["loudness"]["basis"] == "mix"
assert info2["sum_max_error"] < 1e-4

# 3) ceiling: quiet-ish file + loud target -> required gain would clip,
# so the gain must be capped at the ceiling
save_wav(src, np.stack([mix]), sr)
info3 = separate_file(src, out, engine="spectral", repair_opts=no_repair,
                      loudness="custom", loudness_target=-6.5,
                      loudness_basis="mix", loudness_ceiling_db=-1.0)
v3, _ = load_audio(info3["voice_path"])
n3, _ = load_audio(info3["noise_path"])
peak = max(float(np.max(np.abs(v3))), float(np.max(np.abs(n3))))
print(f"ceiling: limited={info3['loudness']['peak_limited']} peak_after={peak:.3f} (ceiling 0.891)")
assert info3["loudness"]["peak_limited"] is True
assert peak <= 0.9
print("OK: post-normalize settings work")
