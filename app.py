"""Voice/Noise Separator — Web + Mac app server (one codebase for both).

Run:  uvicorn app:app --host 0.0.0.0 --port 8200
The Mac launcher (.command) starts this same server and opens it in Safari.
"""
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from core.separator import separate_file

BASE = Path(__file__).resolve().parent
WORK = BASE / "work"
WORK.mkdir(exist_ok=True)

app = FastAPI(title="Voice/Noise Separator")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/engines")
def engines():
    avail = {"spectral": True, "deepfilternet": False, "mossformer2": False,
             "reuse": False, "dbs": False, "bandit": False, "dereverb": False,
             "whisper": False}
    try:
        from core import dbs_engine
        avail["dbs"] = dbs_engine.available()
    except Exception:
        pass
    try:
        from core import bandit_engine
        avail["bandit"] = bandit_engine.available()
    except Exception:
        pass
    try:
        from core import dereverb_engine
        avail["dereverb"] = dereverb_engine.available()
    except Exception:
        avail["dereverb"] = False
    try:
        from core import transcribe
        avail["whisper"] = transcribe.available()
    except Exception:
        avail["whisper"] = False
    try:
        from core import reuse_engine
        avail["reuse"] = reuse_engine.available()
    except Exception:
        pass
    try:
        import torch  # noqa: F401
        from core.separator import _patch_torchaudio_for_dfn
        _patch_torchaudio_for_dfn()  # df 0.5.6 needs the torchaudio shims to import
        import df  # noqa: F401
        avail["deepfilternet"] = True
    except Exception:
        pass
    try:
        import clearvoice  # noqa: F401
        avail["mossformer2"] = True
    except Exception:
        pass
    return avail


@app.post("/api/separate")
async def separate_endpoint(
    file: UploadFile = File(...),
    engine: str = Form("spectral"),
    strength: float = Form(0.65),
    natural: float = Form(0.2),
    fix_clicks: str = Form("true"),
    fix_hum: str = Form("true"),
    fix_dropouts: str = Form("true"),
    fix_wind: str = Form("false"),
    fix_ambience: str = Form("true"),
    preview_start: float = Form(0.0),
    preview_len: float = Form(0.0),
    loudness_preset: str = Form(""),
    loudness_target: str = Form(""),
    loudness_basis: str = Form("mix"),
    loudness_ceiling: str = Form("-1"),
    fix_dereverb: str = Form("false"),
    fix_outdoor: str = Form("false"),
):
    job = WORK / uuid.uuid4().hex[:12]
    job.mkdir(parents=True)
    src = job / Path(file.filename or "input.wav").name
    with src.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    repair_opts = {
        "clicks": fix_clicks == "true",
        "hum": fix_hum == "true",
        "dropouts": fix_dropouts == "true",
        "wind": fix_wind == "true",
        "ambience": fix_ambience == "true",
        "dereverb": fix_dereverb == "true",
        "outdoor": fix_outdoor == "true",
    }
    try:
        info = separate_file(src, job, engine=engine, strength=strength,
                             natural=natural,
                             repair_opts=repair_opts,
                             clip_start=preview_start,
                             clip_len=preview_len if preview_len > 0 else None,
                             loudness=loudness_preset or None,
                             loudness_target=(float(loudness_target)
                                              if loudness_target.strip() else None),
                             loudness_basis=loudness_basis,
                             loudness_ceiling_db=float(loudness_ceiling))
    except Exception as e:  # ffmpeg / decode errors etc. — report plainly
        return {"ok": False, "error": f"処理に失敗しました: {e}"}
    stem_urls = {name: f"/api/download/{job.name}/{Path(p).name}"
                 for name, p in info.get("stems", {}).items()}
    return {
        "ok": True,
        "job": job.name,
        "voice_url": f"/api/download/{job.name}/{Path(info['voice_path']).name}",
        "noise_url": f"/api/download/{job.name}/{Path(info['noise_path']).name}",
        "stem_urls": stem_urls,
        **info,
    }


@app.get("/api/download/{job}/{name}")
def download(job: str, name: str):
    p = (WORK / job / name).resolve()
    if WORK.resolve() not in p.parents or not p.exists():
        return {"ok": False, "error": "not found"}
    return FileResponse(p, filename=p.name, media_type="audio/wav")


@app.post("/api/save_project")
async def save_project(files: list[UploadFile] = File(...)):
    """Save editor-exported stems as a new job (the editor's project save).

    The returned job can be reopened later: its stems are ordinary job
    files, downloadable and loadable into the waveform editor."""
    job = WORK / uuid.uuid4().hex[:12]
    job.mkdir(parents=True)
    saved = {}
    for f in files:
        name = Path(f.filename or "stem.wav").name
        p = job / name
        with p.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved[name] = f"/api/download/{job.name}/{name}"
    return {"ok": True, "job": job.name, "stem_urls": saved}


@app.post("/api/transcribe")
def transcribe_endpoint(job: str = Form(...), name: str = Form(...),
                        model_size: str = Form("small")):
    """Transcribe one stem file of an existing job (Whisper, local)."""
    src = (WORK / job / name).resolve()
    if WORK.resolve() not in src.parents or not src.exists():
        return {"ok": False, "error": "対象ファイルが見つかりません"}
    from core.transcribe import transcribe_file
    out = WORK / job / "transcript"
    try:
        info = transcribe_file(src, out, model_size=model_size)
    except Exception as e:
        return {"ok": False, "error": f"書き出しに失敗しました: {e}"}
    segs = []
    for s in info["segments"]:
        s = dict(s)
        if "clip" in s:
            s["clip_url"] = f"/api/download/{job}/transcript/clips/{Path(s['clip']).name}"
            del s["clip"]
        segs.append(s)
    return {"ok": True,
            "srt_url": f"/api/download/{job}/transcript/{Path(info['srt_path']).name}",
            "segments": segs, "model": info["model"]}


@app.get("/api/download/{job}/transcript/{name}")
def download_transcript(job: str, name: str):
    p = (WORK / job / "transcript" / name).resolve()
    if WORK.resolve() not in p.parents or not p.exists():
        return {"ok": False, "error": "not found"}
    return FileResponse(p, filename=p.name)


@app.get("/api/download/{job}/transcript/clips/{name}")
def download_clip(job: str, name: str):
    p = (WORK / job / "transcript" / "clips" / name).resolve()
    if WORK.resolve() not in p.parents or not p.exists():
        return {"ok": False, "error": "not found"}
    return FileResponse(p, filename=p.name, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)
