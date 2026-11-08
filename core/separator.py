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
  - "dbs"       : optional, DBS-TIGER-DnR (Apache-2.0) — a different mode
                  rather than an enhancement: 3-stem separation of film /
                  video audio into dialogue / sound effects / music (mono,
                  44.1 kHz model). voice = dialogue, extra_stems = sfx/music,
                  noise = residual, so all stems still sum to the original.
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
    # multi-stem engines (dbs): extra stems by name, e.g. {"sfx":..., "music":...}.
    # Invariant then extends: voice + noise + sum(extra_stems) == original (mono).
    extra_stems: dict = field(default_factory=dict)
    # outdoor mode: detected passing-event spans and what was done to each
    events: list = field(default_factory=list)


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


def _separate_dbs(data: np.ndarray, sr: int) -> dict | None:
    """DBS-TIGER-DnR 3-stem separation -> {"speech","sfx","music"} or None."""
    try:
        from . import dbs_engine
        return dbs_engine.separate3(data, sr)
    except Exception:
        return None


def _separate_bandit(data: np.ndarray, sr: int) -> dict | None:
    """BandIt Plus 3-stem separation (stereo) -> {"speech","sfx","music"} or None.

    PERSONAL EDITION ONLY (checkpoint license terms unstated)."""
    try:
        from . import bandit_engine
        return bandit_engine.separate3(data, sr)
    except Exception:
        return None




# ---------------------------------------------------------------- outdoor events
def detect_events(data: np.ndarray, sr: int) -> list[dict]:
    """Detect passing-event spans (train / car / bike / plane / crossing...).

    Frame RMS (100 ms) in dB; baseline = 20th percentile (location bed).
    Frames >= baseline + 10 dB sustained >= 0.7 s become an event; gaps under
    0.8 s are merged. kind_guess is a rough shape-based label only.
    Designed to run on the residual (data - voice) stem: dialogue lands in
    the voice stem, so level alone does not mistake speech for an event.
    """
    mono = to_mono(data)
    win = max(1, int(sr * 0.1))
    n_frames = len(mono) // win
    if n_frames < 10:
        return []
    frames = mono[:n_frames * win].reshape(n_frames, win)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    baseline = float(np.percentile(db, 20))
    hot = db >= baseline + 10.0
    events: list[dict] = []
    i = 0
    while i < n_frames:
        if not hot[i]:
            i += 1
            continue
        last = i
        gap = 0
        j = i
        while j < n_frames:
            if hot[j]:
                last = j
                gap = 0
            else:
                gap += 1
                if gap > 8:
                    break
            j += 1
        dur = (last - i + 1) * 0.1
        if dur >= 0.7:
            peak = float(np.max(db[i:last + 1]) - baseline)
            if dur < 2.0:
                kind = "短い通過・突発音"
            elif dur <= 20.0:
                kind = "通過音（車・電車・バイク・踏切系）"
            else:
                kind = "長い通過音（飛行機・電車系）"
            events.append({"start_sec": round(i * 0.1, 2),
                           "end_sec": round((last + 1) * 0.1, 2),
                           "peak_db": round(peak, 1), "kind_guess": kind})
        i = max(last + 1, i + 1)
    return events


def _apply_outdoor(voice: np.ndarray, data: np.ndarray, sr: int,
                   events: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Segment-adaptive treatment of detected events on the voice stem.

    No-speech event spans are pulled down toward the baseline level
    (aggressive is safe there: nothing spoken can turn fake). Speech event
    spans get one extra strong spectral pass, blended at 70%. The caller
    recomputes noise = data - voice afterwards, so the sum invariant holds.
    """
    if not events:
        return voice, []
    mono = to_mono(data)
    win = max(1, int(sr * 0.1))
    n_frames = len(mono) // win
    if n_frames == 0:
        return voice, []
    frames = mono[:n_frames * win].reshape(n_frames, win)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    baseline_rms = float(np.percentile(rms, 20))
    out = voice.copy()
    fade = max(1, int(sr * 0.05))
    mono_v = to_mono(out)
    mono_r = to_mono(data - out)

    def _speech_present(s: int, e: int) -> bool:
        # Speech check via spectrum shape: in a no-speech span the voice stem
        # holds only leaked event noise, whose average spectrum is nearly
        # identical to the residual's (cosine ~0.99 measured). Dialogue adds
        # harmonic structure the residual lacks (cosine ~0.88-0.91 measured).
        if e - s < 4096:
            return True
        _, _, zv = sp_signal.stft(mono_v[s:e], fs=sr, nperseg=2048, noverlap=1536)
        _, _, zr = sp_signal.stft(mono_r[s:e], fs=sr, nperseg=2048, noverlap=1536)
        va = np.mean(np.abs(zv), axis=1)
        vb = np.mean(np.abs(zr), axis=1)
        cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
        return cos < 0.96

    actions: list[dict] = []
    for ev in events:
        s = int(ev["start_sec"] * sr)
        e = min(int(ev["end_sec"] * sr), out.shape[1])
        if e - s < int(sr * 0.3):
            continue
        seg_voice = float(np.sqrt(np.mean(out[:, s:e] ** 2) + 1e-12))
        if not _speech_present(s, e):
            target = max(baseline_rms, 1e-5)
            gain = float(np.clip(target / (seg_voice + 1e-12), 0.1, 1.0))
            g = np.full(e - s, gain, dtype=np.float32)
            ramp = min(fade, (e - s) // 2)
            if ramp > 0:
                g[:ramp] = np.linspace(1.0, gain, ramp, dtype=np.float32)
                g[-ramp:] = np.linspace(gain, 1.0, ramp, dtype=np.float32)
            out[:, s:e] = (out[:, s:e] * g[None, :]).astype(np.float32)
            actions.append({**ev, "speech": False,
                            "action": f"無音部として強く抑制（×{gain:.2f}）"})
        else:
            pad = int(sr * 0.25)
            s2 = max(0, s - pad)
            e2 = min(out.shape[1], e + pad)
            enh = _enhance_spectral(out[:, s2:e2], sr, 0.9)
            out[:, s2:e2] = (0.3 * out[:, s2:e2] + 0.7 * enh).astype(np.float32)
            actions.append({**ev, "speech": True,
                            "action": "セリフあり：追加の強抑制を70%ブレンド"})
    return out.astype(np.float32), actions

# ---------------------------------------------------------------- public API
# Naturalness-first defaults (Kanta 2026-11-06: "a removal that does not
# sound fake matters more than maximum removal"). Strength is kept moderate
# and a slice of the removed sound is mixed back into the voice stem so AI
# engines do not sound watery / metallic / gated. Max removal is opt-in.
DEFAULT_STRENGTH = 0.65
DEFAULT_NATURAL = 0.20

DEFAULT_REPAIR = {"clicks": True, "hum": True, "dropouts": True, "wind": False,
                  "ambience": True}


def separate(data: np.ndarray, sr: int, engine: str = "spectral",
             strength: float = DEFAULT_STRENGTH, natural: float = DEFAULT_NATURAL,
             repair_opts: dict | None = None) -> SeparationResult:
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
    want_dereverb = bool(opts.pop("dereverb", False))  # not a repair.py option
    want_outdoor = bool(opts.pop("outdoor", False))  # applied after separation
    if any(opts.values()):
        from .repair import repair as run_repair
        work, repair_report = run_repair(data, sr, sensitivity=1.0, **opts)
        notes.append(f"repair: {repair_report}")
    if want_dereverb:
        try:
            from . import dereverb_engine
            work = dereverb_engine.process(work, sr)
            notes.append("dereverb: AI reverb removal applied (MelBand RoFormer, personal edition)")
        except Exception:
            notes.append("dereverb is not installed in this environment; skipped.")

    if engine == "dbs":
        stems = _separate_dbs(work, sr)
        if stems is not None:
            # DnR is mono: the invariant is kept against the mono downmix.
            # voice = dialogue, extra stems = sfx / music, and "noise" is the
            # residual so voice + sfx + music + noise == original mono exactly.
            mono_orig = to_mono(data)[None, :].astype(np.float32)
            speech = stems["speech"].astype(np.float32)
            extra = {"sfx": stems["sfx"].astype(np.float32),
                     "music": stems["music"].astype(np.float32)}
            residual = (mono_orig - speech - extra["sfx"] - extra["music"]).astype(np.float32)
            nat = float(np.clip(natural, 0.0, 0.9))
            if nat > 0:
                speech = (speech + nat * residual).astype(np.float32)
                residual = ((1.0 - nat) * residual).astype(np.float32)
                notes.append(f"natural: ambience return {nat:.2f} from residual (naturalness priority)")
            notes.append("dbs: 3-stem separation (dialogue / sfx / music), mono output")
            return SeparationResult(voice=speech, noise=residual, sr=sr,
                                    engine_used="dbs", engine_requested=engine,
                                    notes=notes, extra_stems=extra)
        notes.append("dbs is not installed in this environment; "
                     "fell back to the built-in spectral engine.")

    if engine == "bandit":
        stems = _separate_bandit(work, sr)
        if stems is not None:
            # Stereo 3-stem: voice = dialogue, extra = sfx / music, and
            # "noise" is the residual so all stems sum back to the original.
            speech = stems["speech"].astype(np.float32)
            extra = {"sfx": stems["sfx"].astype(np.float32),
                     "music": stems["music"].astype(np.float32)}
            residual = (data - speech - extra["sfx"] - extra["music"]).astype(np.float32)
            nat = float(np.clip(natural, 0.0, 0.9))
            if nat > 0:
                speech = (speech + nat * residual).astype(np.float32)
                residual = ((1.0 - nat) * residual).astype(np.float32)
                notes.append(f"natural: ambience return {nat:.2f} from residual (naturalness priority)")
            notes.append("bandit: 3-stem separation (dialogue / sfx / music), stereo")
            return SeparationResult(voice=speech, noise=residual, sr=sr,
                                    engine_used="bandit", engine_requested=engine,
                                    notes=notes, extra_stems=extra)
        notes.append("bandit is not installed in this environment; "
                     "fell back to the built-in spectral engine.")

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
    # Naturalness blend: mix a slice of what was removed back into the voice.
    # This is the only control AI engines get (they ignore `strength`), and it
    # keeps voice + noise == original exactly, so nothing is invented.
    nat = float(np.clip(natural, 0.0, 0.9))
    if nat > 0:
        voice = (voice + nat * (data - voice)).astype(np.float32)
        notes.append(f"natural: ambience return {nat:.2f} (removed sound mixed back; naturalness priority)")
    ev_actions: list[dict] = []
    if want_outdoor:
        voice, ev_actions = _apply_outdoor(voice, data, sr, detect_events((data - voice), sr))
        if ev_actions:
            for a in ev_actions:
                notes.append(f"outdoor: {a['start_sec']}-{a['end_sec']}s "
                             f"+{a['peak_db']}dB {a['kind_guess']} -> {a['action']}")
        else:
            notes.append("outdoor: no passing events detected")
    noise = (data - voice).astype(np.float32)
    return SeparationResult(voice=voice, noise=noise, sr=sr,
                            engine_used=used, engine_requested=engine, notes=notes,
                            events=ev_actions)


def separate_file(in_path: str | Path, out_dir: str | Path,
                  engine: str = "spectral", strength: float = DEFAULT_STRENGTH,
                  natural: float = DEFAULT_NATURAL,
                  repair_opts: dict | None = None,
                  clip_start: float = 0.0, clip_len: float | None = None,
                  loudness: str | None = None) -> dict:
    """File-in / files-out. Returns paths + metadata. Used by Web, Mac and CLI.

    clip_start / clip_len (seconds) implement the trial-separation preview:
    when clip_len is given, only that slice (at most 60 s) is separated so a
    long file can be auditioned before committing to a full run. The result
    metadata then carries a "preview" block with the slice position."""
    in_path, out_dir = Path(in_path), Path(out_dir)
    data, sr = load_audio(in_path)
    if sr != TARGET_SR:
        data, sr = resample(data, sr)
        data = data.astype(np.float32)
    preview = None
    if clip_len is not None:
        full_n = data.shape[1]
        start = int(min(max(clip_start, 0.0), full_n / sr) * sr)
        length = int(min(max(clip_len, 0.1), 60.0) * sr)
        data = data[:, start:start + length]
        preview = {"start_sec": round(start / sr, 3),
                   "length_sec": round(data.shape[1] / sr, 3),
                   "full_duration_sec": round(full_n / sr, 3)}
    result = separate(data, sr, engine=engine, strength=strength, natural=natural, repair_opts=repair_opts)
    loudness_info = None
    if loudness:
        from .loudness import PRESETS, gain_for_target
        if loudness in PRESETS:
            gain_db, measured, limited = gain_for_target(data, sr, PRESETS[loudness])
            lin = float(10 ** (gain_db / 20.0))
            result.voice = (result.voice * lin).astype(np.float32)
            result.noise = (result.noise * lin).astype(np.float32)
            result.extra_stems = {k: (v * lin).astype(np.float32)
                                  for k, v in result.extra_stems.items()}
            data = (data * lin).astype(np.float32)  # invariant check vs gained original
            loudness_info = {"preset": loudness, "target_lufs": PRESETS[loudness],
                             "measured_lufs": round(measured, 2),
                             "gain_db": round(gain_db, 2), "peak_limited": limited}
    stem = in_path.stem
    voice_path = out_dir / f"{stem}_VOICE.wav"
    noise_path = out_dir / f"{stem}_NOISE.wav"
    save_wav(voice_path, result.voice, sr)
    save_wav(noise_path, result.noise, sr)
    stems_info = {"voice": str(voice_path), "noise": str(noise_path)}
    extra_paths = {}
    for name, arr in result.extra_stems.items():
        p = out_dir / f"{stem}_{name.upper()}.wav"
        save_wav(p, arr, sr)
        extra_paths[name] = str(p)
        stems_info[name] = str(p)
    total = result.voice + result.noise
    for arr in result.extra_stems.values():
        total = total + arr
    # mono multi-stem engines (dbs) are checked against the mono downmix;
    # stereo ones (bandit) and the classic 2-stem path against the input.
    ref = data if total.shape == data.shape else to_mono(data)[None, :]
    return {
        "voice_path": str(voice_path),
        "noise_path": str(noise_path),
        "extra_paths": extra_paths,
        "stems": stems_info,
        "sample_rate": sr,
        "channels": int(data.shape[0]),
        "duration_sec": round(data.shape[1] / sr, 3),
        "engine_used": result.engine_used,
        "engine_requested": result.engine_requested,
        "strength": strength,
        "natural": natural,
        "preview": preview,
        "loudness": loudness_info,
        "notes": result.notes,
        "events": result.events,
        # integrity check for the editor: stems must sum back to the original
        # (for multi-stem engines the reference is the mono downmix)
        "sum_max_error": float(np.max(np.abs(total - ref))),
    }


if __name__ == "__main__":  # CLI: python -m core.separator in.wav out_dir [strength]
    import json, sys
    strength = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_STRENGTH
    print(json.dumps(separate_file(sys.argv[1], sys.argv[2], strength=strength), indent=2))
