# -*- coding: utf-8 -*-
"""
VoiceNoiseSeparate.py — DaVinci Resolve Studio script

Selected timeline clips -> local Voice/Noise Separator server -> the VOICE
and NOISE stems come back onto two new audio tracks at the same positions.

Install: copy this file into the Resolve scripts folder and run it from
  Workspace menu > Scripts > Edit > VoiceNoiseSeparate
  macOS: ~/Library/Application Support/Blackmagic Design/DaVinci Resolve/
         Support/Fusion/Scripts/Edit/
  Windows: %APPDATA%/Blackmagic Design/DaVinci Resolve/Support/Fusion/
           Scripts/Edit/

Requirements: DaVinci Resolve STUDIO (scripting API), Resolve 21.0.3+
recommended (Timeline.GetSelectedClips), ffmpeg on PATH, and the
Voice/Noise Separator app running locally (audio portal or standalone).

v1 limitations: retimed/speed-changed clips and nested timelines are not
handled (source in/out are taken at the source media frame rate).
"""
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
import uuid

# ------------------------------------------------------------------ config
CONFIG = {
    "server": "http://127.0.0.1:8200",   # separator app (portal or standalone)
    "engine": "spectral",                # spectral / deepfilternet / mossformer2 / reuse / dbs / bandit
    "strength": 0.65,
    "natural": 0.20,
    "fix_outdoor": True,
    "fix_ambience": True,
    "loudness_preset": "",               # "" / streaming / podcast / broadcast / cinema / custom
    "ffmpeg": "ffmpeg",
    "dry_run": False,                    # True: print the plan, change nothing
}
_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "voice_noise_config.json")
if os.path.exists(_cfg_path):
    try:
        with open(_cfg_path, "r", encoding="utf-8") as f:
            CONFIG.update(json.load(f))
    except Exception as e:
        print("[VoiceNoise] config load failed, using defaults:", e)


def log(msg):
    print("[VoiceNoise] " + str(msg))


# ------------------------------------------------------------------ server
def _multipart(fields, file_field, file_path):
    boundary = "----voicenoise" + uuid.uuid4().hex
    body = b""
    for name, value in fields.items():
        body += ("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                 % (boundary, name, value)).encode("utf-8")
    with open(file_path, "rb") as f:
        data = f.read()
    body += ("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n"
             "Content-Type: audio/wav\r\n\r\n"
             % (boundary, file_field, os.path.basename(file_path))).encode("utf-8")
    body += data + ("\r\n--%s--\r\n" % boundary).encode("utf-8")
    return body, "multipart/form-data; boundary=%s" % boundary


def separate_via_server(wav_path, workdir):
    """POST one wav to the separator and download the stems. Returns
    (voice_path, noise_path). Raises on failure."""
    fields = {
        "engine": CONFIG["engine"],
        "strength": str(CONFIG["strength"]),
        "natural": str(CONFIG["natural"]),
        "fix_clicks": "false", "fix_hum": "true", "fix_dropouts": "false",
        "fix_wind": "false",
        "fix_ambience": "true" if CONFIG["fix_ambience"] else "false",
        "fix_outdoor": "true" if CONFIG["fix_outdoor"] else "false",
        "fix_dereverb": "false",
        "loudness_preset": CONFIG["loudness_preset"],
    }
    body, ctype = _multipart(fields, "file", wav_path)
    req = urllib.request.Request(CONFIG["server"] + "/api/separate",
                                 data=body, headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        info = json.loads(resp.read().decode("utf-8"))
    if not info.get("ok"):
        raise RuntimeError(info.get("error") or "separation failed")
    out = {}
    for key, url_key in (("voice", "voice_url"), ("noise", "noise_url")):
        url = info.get(url_key)
        if not url:
            raise RuntimeError("server response missing " + url_key)
        dest = os.path.join(workdir, os.path.basename(url))
        with urllib.request.urlopen(CONFIG["server"] + url, timeout=600) as r, \
                open(dest, "wb") as f:
            f.write(r.read())
        out[key] = dest
    return out["voice"], out["noise"]


def extract_segment(src_path, start_sec, dur_sec, dest):
    cmd = [CONFIG["ffmpeg"], "-y", "-ss", "%.3f" % start_sec, "-i", src_path,
           "-t", "%.3f" % dur_sec, "-ar", "48000", "-vn", dest]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(dest):
        raise RuntimeError("ffmpeg extract failed: " + (r.stderr or "")[-400:])


def wav_frames(path, fps):
    import wave
    with wave.open(path, "rb") as w:
        return int(round(w.getnframes() / float(w.getframerate()) * fps))


# ------------------------------------------------------------------ resolve
def get_resolve():
    try:
        import DaVinciResolveScript as dvr
        return dvr.scriptapp("Resolve")
    except Exception:
        return None


def collect_audio_items(timeline):
    """Selected audio items; video selections contribute their linked audio.
    Falls back to the clip under the playhead on old Resolve versions."""
    items = []
    if hasattr(timeline, "GetSelectedClips"):
        selected = timeline.GetSelectedClips() or []
    else:
        log("GetSelectedClips not available (Resolve < 21.0.3?) - "
            "using the clip under the playhead")
        cur = timeline.GetCurrentVideoItem()
        selected = [cur] if cur else []
    seen = set()
    for it in selected:
        cands = [it]
        try:
            cands += it.GetLinkedItems() or []
        except Exception:
            pass
        for c in cands:
            try:
                ttype, _ = c.GetTrackTypeAndIndex()
            except Exception:
                continue
            if ttype != "audio":
                continue
            key = (c.GetMediaPoolItem().GetClipProperty("File Path")
                   if c.GetMediaPoolItem() else None,
                   c.GetSourceStartFrame(), c.GetSourceEndFrame())
            if key in seen or key[0] is None:
                continue
            seen.add(key)
            items.append(c)
    return items


def main():
    resolve = get_resolve()
    if resolve is None:
        log("DaVinci Resolve scripting is not available. "
            "Run from inside Resolve Studio (Workspace > Scripts).")
        return
    pm = resolve.GetProjectManager()
    project = pm.GetCurrentProject()
    timeline = project.GetCurrentTimeline() if project else None
    if timeline is None:
        log("No current timeline.")
        return
    fps = float(timeline.GetSetting("timelineFrameRate"))
    items = collect_audio_items(timeline)
    if not items:
        log("No audio clips selected.")
        return
    log("%d clip(s) to process (fps=%s, dry_run=%s)"
        % (len(items), fps, CONFIG["dry_run"]))

    mediapool = project.GetMediaPool()
    workdir = tempfile.mkdtemp(prefix="voicenoise_")
    placed = 0
    track_idx = {}
    for item in items:
        mpi = item.GetMediaPoolItem()
        src_path = mpi.GetClipProperty("File Path")
        try:
            src_fps = float(mpi.GetClipProperty("FPS") or fps)
        except Exception:
            src_fps = fps
        start_sec = item.GetSourceStartFrame() / src_fps
        dur_sec = (item.GetSourceEndFrame() - item.GetSourceStartFrame()) / src_fps
        name = os.path.splitext(os.path.basename(src_path))[0]
        seg = os.path.join(workdir, name + "_segment.wav")
        log("clip %s [%.2fs +%.2fs]" % (name, start_sec, dur_sec))
        if CONFIG["dry_run"]:
            continue
        try:
            extract_segment(src_path, start_sec, dur_sec, seg)
            voice_path, noise_path = separate_via_server(seg, workdir)
        except Exception as e:
            log("FAILED on %s: %s" % (name, e))
            continue
        for label, path in (("VOICE", voice_path), ("NOISE", noise_path)):
            if label not in track_idx:
                timeline.AddTrack("audio")
                idx = timeline.GetTrackCount("audio")
                timeline.SetTrackName("audio", idx, "VoiceNoise " + label)
                track_idx[label] = idx
            imported = mediapool.ImportMedia([path])
            if not imported:
                log("ImportMedia failed for " + path)
                continue
            frames = wav_frames(path, fps)
            res = mediapool.AppendToTimeline([{
                "mediaPoolItem": imported[0], "mediaType": 2,
                "trackIndex": track_idx[label],
                "recordFrame": item.GetStart(),
                "startFrame": 0, "endFrame": frames,
            }])
            if res:
                placed += 1
    log("done: %d stem clip(s) placed" % placed)


if __name__ == "__main__" or True:
    # Resolve runs menu scripts as __main__-less modules; call unconditionally
    # but only when the Resolve API is reachable, so importing this file in a
    # plain Python (for tests) is harmless.
    if get_resolve() is not None:
        main()
