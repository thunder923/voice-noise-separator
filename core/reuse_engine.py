"""NVIDIA NemotronLabs RE-USE engine adapter.

RE-USE (nvidia/NVIDIA-NemotronLabs-RE-USE, OpenMDW-1.1) is a 9.6M-parameter
universal speech enhancement model: it removes additive noise, reverb,
clipping, bandwidth limits, codec artifacts and packet loss in one pass,
at any sampling rate from 8 to 48 kHz (sampling-frequency-independent STFT).

Integration notes:
  - Model files are NOT vendored in this repo. Setup downloads the official
    Hugging Face snapshot (code + weights, ~38 MB) into
    checkpoints/NVIDIA-NemotronLabs-RE-USE/ and this module loads the model
    from there. If the folder is missing, the engine reports unavailable.
  - The official inference script is CUDA-only and the model code imports
    mamba_ssm, which has no CPU / macOS wheels. When mamba_ssm is not
    importable, core.mamba_cpu installs a pure-PyTorch Mamba with identical
    parameters and math (verified against the official reference scan).
  - Processing follows the official long-file recipe (inference_chunk.py):
    per channel, 5 s chunks, 50% hop, Hann-window overlap-add, plus the
    sweep-artifact suppression step from inference.py.
  - On CPU this engine is SLOW (tens of times slower than real time at
    48 kHz on a small VM). Treat it as a quality-first option for short
    clips; MossFormer2 stays the practical default for long material.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

CKPT = (Path(__file__).resolve().parent.parent
        / "checkpoints" / "NVIDIA-NemotronLabs-RE-USE")
CHUNK_SEC = 5.0
HOP_PORTION = 0.5

_model = None
_cfg = None
_device = None
_load_error: str | None = None


def available() -> bool:
    """True when the model snapshot is downloaded and torch can load."""
    if not (CKPT / "model.safetensors").exists():
        return False
    try:
        import torch  # noqa: F401
        import safetensors  # noqa: F401
    except Exception:
        return False
    return True


def _pick_device(torch) -> str:
    forced = os.environ.get("VNS_REUSE_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load():
    global _model, _cfg, _device, _load_error
    if _model is not None:
        return _model
    if _load_error is not None:
        raise RuntimeError(_load_error)
    try:
        import torch
        try:
            import mamba_ssm  # noqa: F401  (real package wins when present)
        except Exception:
            from . import mamba_cpu
            mamba_cpu.install()
        if str(CKPT) not in sys.path:
            sys.path.insert(0, str(CKPT))
        import importlib
        SEMamba = importlib.import_module(
            "models.generator_SEMamba_time_d4").SEMamba
        from safetensors.torch import load_file

        _cfg = json.loads((CKPT / "config.json").read_text(encoding="utf-8"))
        _device = _pick_device(torch)
        model = SEMamba(_cfg)
        missing, unexpected = model.load_state_dict(
            load_file(str(CKPT / "model.safetensors")), strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch: missing={missing[:3]} unexpected={unexpected[:3]}")
        _model = model.to(_device).eval()
        return _model
    except Exception as e:  # remember and re-raise so callers can fall back
        _load_error = f"RE-USE load failed: {e}"
        raise


def _make_even(v: float) -> int:
    v = int(round(v))
    return v if v % 2 == 0 else v + 1


def _enhance_mono(x: np.ndarray, sr: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    model = _load()  # also puts the snapshot dir on sys.path for the imports below
    from models.stfts import mag_phase_stft, mag_phase_istft
    from utils.util import pad_or_trim_to_match
    st = _cfg["stft_cfg"]
    base = st["sampling_rate"]
    n_fft = _make_even(st["n_fft"] * sr // base)
    hop = _make_even(st["hop_size"] * sr // base)
    win = _make_even(st["win_size"] * sr // base)
    cf = _cfg["model_cfg"]["compress_factor"]

    wav = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None, :].to(_device)
    chunk = int(CHUNK_SEC * sr)
    hop_len = int(HOP_PORTION * chunk)
    window = torch.clamp(torch.hann_window(chunk), min=1e-7).to(_device)
    out = torch.zeros_like(wav)
    wsum = torch.zeros_like(wav)
    n = wav.shape[1]
    n_chunks = max(1, int(np.ceil((n - chunk) / hop_len)) + 1) if n > chunk else 1
    with torch.no_grad():
        for i in range(n_chunks):
            if n <= chunk:
                seg, start = wav, 0
            elif i < n_chunks - 1:
                start = i * hop_len
                seg = wav[:, start:start + chunk]
            else:
                start = n - chunk
                seg = wav[:, start:]
            mag, pha, _ = mag_phase_stft(seg, n_fft=n_fft, hop_size=hop,
                                         win_size=win, compress_factor=cf,
                                         center=True, addeps=False)
            amp_g, pha_g, _ = model(mag, pha)
            # official sweep-artifact suppression (inference.py)
            magv = torch.expm1(F.relu(amp_g))
            zero_portion = torch.sum(magv == 0, 1) / magv.shape[1]
            amp_g[:, :, (zero_portion > 0.5)[0]] = 0
            audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop, win, cf)
            audio_g = pad_or_trim_to_match(seg, audio_g, pad_value=1e-8)
            out[:, start:start + seg.shape[1]] += audio_g * window[:audio_g.shape[1]]
            wsum[:, start:start + seg.shape[1]] += window[:audio_g.shape[1]]
    return (out / wsum).cpu().numpy()[0].astype(np.float32)


def enhance(data: np.ndarray, sr: int) -> np.ndarray | None:
    """Enhance [ch, samples] float32 audio. Returns None on any failure so
    the caller can fall back to the spectral engine with a note."""
    if not available():
        return None
    try:
        chans = [_enhance_mono(data[c], sr) for c in range(data.shape[0])]
        out = np.stack(chans).astype(np.float32)
        if out.shape != data.shape:
            m = min(out.shape[1], data.shape[1])
            out = np.pad(out[:, :m], ((0, 0), (0, data.shape[1] - m)))
        return out
    except Exception:
        return None
