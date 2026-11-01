"""
Voice / Noise separator core — shared by Web, Mac app and (algorithm reference) plugin.

Principle (important for film dialogue editing):
    voice  = AI / spectral enhanced speech estimate
    noise  = original - voice        (phase-aligned residual)

Because noise is the *residual*,  voice + noise == original  (within float error).
An editor can therefore put both stems on two tracks in Pro Tools, and by
simply moving the two faders re-balance dialogue vs. ambience — nothing is lost.

Engines:
  - "spectral"  : built-in, no model download, numpy/scipy only.
                  Estimates a per-frequency noise profile from low-energy
                  (non-speech) frames, builds a soft Wiener-style mask,
                  strength-controlled. Always available, also used as fallback.
  - "deepfilternet" : optional, if the `deepfilternet` package + torch are
                  installed. Low-complexity, real-time capable, 48 kHz.
  - "mossformer2"   : optional, if `clearvoice` (ClearerVoice-Studio) is
                  installed. Highest quality offline model
                  (MossFormer2_SE_48K, PESQ 3.15 / SI-SDR 19.38 in the paper),
                  best suited for final film dialogue cleanup.
  - "reuse"     : optional, NVIDIA NemotronLabs RE-USE (OpenMDW-1.1),
                  a 9.6M-parameter universal speech enhancement model that
                  also repairs reverb / clipping / codec artifacts, if the
                  model snapshot is downloaded into checkpoints/ (see
                  core/reuse_engine.py). Quality-first but slow on CPU.
  If an optional engine is requested but not installed, we fall back to
  "spectral" and report that in the result metadata — never fail silently.
"""
from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal

TARGET_SR = 48000  # film / broadcast standard


# ---------------------------------------------------------------- io helpers
def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load any audio/video file as float32 mono-or-stereo [channels, samples] at native SR.
    WAV is read natively; everything else goes through ffmpeg."""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            return _load_wav(path)
        except Exception:
            pass  # fall through to ffmpeg (e.g. 24-bit / float WAV variants)
    return _load_via_ffmpeg(path)


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        data = v.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    data = data.reshape(-1, ch).T  # [ch, samples]
    return data, sr


def _load_via_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(TARGET_SR), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    data = np.frombuffer(out, dtype=np.float32).reshape(-1, 2).T.copy()
    return data, TARGET_SR


def save_wav(path: str | Path, data: np.ndarray, sr: int) -> None:
    """Save float [ch, samples] as 24-bit WAV (film delivery standard)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(data, -1.0, 1.0)
    ints = (clipped.T * 8388607.0).astype(np.int32)
    b = np.empty((ints.shape[0], ints.shape[1], 3), dtype=np.uint8)
    b[:, :, 0] = ints & 0xFF
    b[:, :, 1] = (ints >> 8) & 0xFF
    b[:, :, 2] = (ints >> 16) & 0xFF
    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[0])
        w.setsampwidth(3)
        w.setframerate(sr)
        w.writeframes(b.tobytes())


def to_mono(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=0) if data.shape[0] > 1 else data[0]


def resample(data: np.ndarray, sr: int, target: int = TARGET_SR) -> tuple[np.ndarray, int]:
    if sr == target:
        return data, sr
    n_new = int(round(data.shape[1] * target / sr))
    out = np.stack([sp_signal.resample_poly(ch, target, sr)[:n_new] if False else
                    sp_signal.resample(ch, n_new) for ch in data])
    return out.astype(np.float32), target


# ---------------------------------------------------------------- result
@dataclass
class SeparationResult:
    voice: np.ndarray          # [ch, samples] float32
    noise: np.ndarray          # [ch, samples] float32  (original - voice)
    sr: int
    engine_used: str
    engine_requested: str
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- spectral engine
def _spectral_enhance_channel(x: np.ndarray, sr: int, strength: float,
                              noise_percentile: float = 15.0) -> np.ndarray:
    """Soft-mask enhancement for one channel.

    strength 0..1 : 0 = passthrough, 1 = maximum suppression.
    Noise profile = low percentile of per-frequency magnitude over time
    (dialogue recordings always contain pauses / room tone to learn from),
    refined with a speech-presence smoothing so fricatives are not eaten.
    """
    nperseg, noverlap = 2048, 1536
    f, t, Z = sp_signal.stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Z)

    # Time-varying noise profile via block minimum-statistics:
    # film location sound is non-stationary (wind gusts, passing cars), so a
    # single global profile fails. Split time into ~0.7 s blocks, take the
    # per-frequency 20th percentile in each block (x1.9 Rayleigh correction:
    # mean / p20 of a Rayleigh distribution ~= 1.9), then smooth across blocks.
    from scipy.ndimage import uniform_filter1d
    W = 64
    n_blocks = (mag.shape[1] + W - 1) // W
    noise_mag = np.zeros_like(mag)
    for i in range(n_blocks):
        blk = mag[:, i * W:(i + 1) * W]
        est = np.percentile(blk, 20.0, axis=1, keepdims=True) * 1.9
        noise_mag[:, i * W:(i + 1) * W] = est
    noise_mag = uniform_filter1d(noise_mag, size=W, axis=1, mode="nearest")
    noise_mag = np.maximum(noise_mag, 1e-10)
    # a-priori SNR estimate -> Wiener-style soft mask
    snr = np.maximum(mag / noise_mag - 1.0, 0.0)
    mask = snr / (snr + 1.0)                       # 0..1 speech presence
    # strength interpolates between no suppression and full mask;
    # floor keeps a little ambience so dialogue does not sound "gated to death"
    floor = 1.0 - 0.97 * strength
    gain = floor + (1.0 - floor) * np.power(mask, 0.5 + 1.5 * strength)
    # smooth in time/freq to avoid musical noise
    gain = sp_signal.medfilt(gain, kernel_size=(1, 5))
    Z_enh = Z * gain
    _, y = sp_signal.istft(Z_enh, fs=sr, nperseg=nperseg, noverlap=noverlap)
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y[:len(x)].astype(np.float32)


def _enhance_spectral(data: np.ndarray, sr: int, strength: float) -> np.ndarray:
    return np.stack([_spectral_enhance_channel(ch, sr, strength) for ch in data])


# ---------------------------------------------------------------- optional AI engines
def _via_temp_wav(data: np.ndarray, sr: int, fn) -> np.ndarray | None:
    """Run a file-based AI engine on a temp WAV and read its output back."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "in.wav"
        dst = td / "out.wav"
        save_wav(src, data, sr)
        try:
            fn(src, dst)
        except Exception:
            return None
        if not dst.exists():
            return None
        out, osr = load_audio(dst)
        if osr != sr:
            out, _ = resample(out, osr, sr)
        return out.astype(np.float32)


def _patch_torchaudio_for_dfn() -> None:
    """DeepFilterNet 0.5.6 targets old torchaudio: torchaudio>=2.9 removed
    torchaudio.backend / torchaudio.info, and torchaudio.load now needs
    torchcodec. Shim those three with soundfile equivalents (WAV-only,
    same contracts). Verified against torchaudio 2.11.0+cpu."""
    import dataclasses
    import sys
    import types

    import torchaudio

    @dataclasses.dataclass
    class AudioMetaData:
        sample_rate: int = 0
        num_frames: int = 0
        num_channels: int = 0
        bits_per_sample: int = 0
        encoding: str = ""

    if "torchaudio.backend.common" not in sys.modules:
        backend = types.ModuleType("torchaudio.backend")
        common = types.ModuleType("torchaudio.backend.common")
        common.AudioMetaData = AudioMetaData
        backend.common = common
        sys.modules["torchaudio.backend"] = backend
        sys.modules["torchaudio.backend.common"] = common
    if not hasattr(torchaudio, "info"):
        import soundfile as sf

        def _info(file, **kw):
            i = sf.info(file)
            return AudioMetaData(sample_rate=i.samplerate, num_frames=i.frames,
                                 num_channels=i.channels, bits_per_sample=16,
                                 encoding="PCM_S")
        torchaudio.info = _info
    try:
        import torchcodec  # noqa: F401
    except Exception:
        import soundfile as sf
        import torch

        def _load(file, frame_offset=0, num_frames=-1, **kw):
            arr, rate = sf.read(str(file), dtype="float32", always_2d=True)
            arr = arr.T  # [ch, samples]
            if frame_offset:
                arr = arr[:, frame_offset:]
            if num_frames and num_frames > 0:
                arr = arr[:, :num_frames]
            return torch.from_numpy(np.ascontiguousarray(arr)), rate
        torchaudio.load = _load


def _enhance_deepfilternet(data: np.ndarray, sr: int) -> np.ndarray | None:
    try:
        import torch
        _patch_torchaudio_for_dfn()
        from df.enhance import enhance, init_df, load_audio as df_load
    except Exception:
        return None

    def run(src: Path, dst: Path) -> None:
        model, df_state, _ = init_df()
        audio, _ = df_load(str(src), sr=df_state.sr())
        enhanced = enhance(model, df_state, audio)
        import soundfile as sf
        sf.write(str(dst), enhanced.squeeze(0).numpy().T if enhanced.dim() > 1
                 else enhanced.numpy(), df_state.sr())

    return _via_temp_wav(data, sr, run)


def _enhance_mossformer2(data: np.ndarray, sr: int) -> np.ndarray | None:
    try:
        from clearvoice import ClearVoice
    except Exception:
        return None

    def run(src: Path, dst: Path) -> None:
        import shutil
        cv = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
        out = cv(input_path=str(src), online_write=False)
        if isinstance(out, (str, Path)) and Path(out).exists():
            shutil.copy(out, dst)
        else:  # array return: ClearVoice gives [ch, samples]; soundfile wants [samples, ch]
            import soundfile as sf
            arr = np.asarray(out, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr.T if arr.shape[0] <= 8 else arr
            sf.write(str(dst), arr, sr)

    return _via_temp_wav(data, sr, run)


def _enhance_reuse(data: np.ndarray, sr: int) -> np.ndarray | None:
    try:
        from . import reuse_engine
        return reuse_engine.enhance(data, sr)
    except Exception:
        return None


# ---------------------------------------------------------------- public API
DEFAULT_REPAIR = {"clicks": True, "hum": True, "dropouts": True, "wind": False,
                  "ambience": True}


def separate(data: np.ndarray, sr: int, engine: str = "spectral",
             strength: float = 0.8, repair_opts: dict | None = None) -> SeparationResult:
    """Separate [ch, samples] float audio into voice / noise stems.

    Pipeline: targeted repair (clicks / hum / dropouts / wind) -> enhancement
    engine -> noise = ORIGINAL - voice, so repairs are part of the voice stem
    and everything removed stays audible in the noise stem."""
    notes: list[str] = []
    used = "spectral"
    voice = None

    work = data
    repair_report: dict = {}
    opts = {**DEFAULT_REPAIR, **(repair_opts or {})}
    if any(opts.values()):
        from .repair import repair as run_repair
        work, repair_report = run_repair(data, sr, sensitivity=1.0, **opts)
        notes.append(f"repair: {repair_report}")

    if engine in ("deepfilternet", "mossformer2", "reuse"):
        fn = {"deepfilternet": _enhance_deepfilternet,
              "mossformer2": _enhance_mossformer2,
              "reuse": _enhance_reuse}[engine]
        voice = fn(work, sr)
        if voice is not None:
            used = engine
        else:
            notes.append(f"{engine} is not installed in this environment; "
                         f"fell back to the built-in spectral engine.")
    if voice is None:
        voice = _enhance_spectral(work, sr, float(np.clip(strength, 0.0, 1.0)))

    voice = voice.astype(np.float32)
    if voice.shape != data.shape:  # align length exactly so residual is sample-accurate
        n = min(voice.shape[1], data.shape[1])
        voice = np.pad(voice[:, :n], ((0, 0), (0, data.shape[1] - n)))
    noise = (data - voice).astype(np.float32)
    return SeparationResult(voice=voice, noise=noise, sr=sr,
                            engine_used=used, engine_requested=engine, notes=notes)


def separate_file(in_path: str | Path, out_dir: str | Path,
                  engine: str = "spectral", strength: float = 0.8,
                  repair_opts: dict | None = None) -> dict:
    """File-in / files-out. Returns paths + metadata. Used by Web, Mac and CLI."""
    in_path, out_dir = Path(in_path), Path(out_dir)
    data, sr = load_audio(in_path)
    if sr != TARGET_SR:
        data, sr = resample(data, sr)
        data = data.astype(np.float32)
    result = separate(data, sr, engine=engine, strength=strength, repair_opts=repair_opts)
    stem = in_path.stem
    voice_path = out_dir / f"{stem}_VOICE.wav"
    noise_path = out_dir / f"{stem}_NOISE.wav"
    save_wav(voice_path, result.voice, sr)
    save_wav(noise_path, result.noise, sr)
    return {
        "voice_path": str(voice_path),
        "noise_path": str(noise_path),
        "sample_rate": sr,
        "channels": int(data.shape[0]),
        "duration_sec": round(data.shape[1] / sr, 3),
        "engine_used": result.engine_used,
        "engine_requested": result.engine_requested,
        "notes": result.notes,
        # integrity check for the editor: stems must sum back to the original
        "sum_max_error": float(np.max(np.abs((result.voice + result.noise) - data))),
    }


if __name__ == "__main__":  # CLI: python -m core.separator in.wav out_dir [strength]
    import json, sys
    strength = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    print(json.dumps(separate_file(sys.argv[1], sys.argv[2], strength=strength), indent=2))
