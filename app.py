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
    avail = {"spectral": True, "deepfilternet": False, "mossformer2": False}
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
    strength: float = Form(0.8),
    fix_clicks: str = Form("true"),
    fix_hum: str = Form("true"),
    fix_dropouts: str = Form("true"),
    fix_wind: str = Form("false"),
    fix_ambience: str = Form("true"),
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
    }
    try:
        info = separate_file(src, job, engine=engine, strength=strength,
                             repair_opts=repair_opts)
    except Exception as e:  # ffmpeg / decode errors etc. — report plainly
        return {"ok": False, "error": f"処理に失敗しました: {e}"}
    return {
        "ok": True,
        "job": job.name,
        "voice_url": f"/api/download/{job.name}/{Path(info['voice_path']).name}",
        "noise_url": f"/api/download/{job.name}/{Path(info['noise_path']).name}",
        **info,
    }


@app.get("/api/download/{job}/{name}")
def download(job: str, name: str):
    p = (WORK / job / name).resolve()
    if WORK.resolve() not in p.parents or not p.exists():
        return {"ok": False, "error": "not found"}
    return FileResponse(p, filename=p.name, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)
