"""
Audio Visualizer — Backend
FastAPI server with username/password session auth.
"""

import os
import uuid
import logging
import asyncio
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import librosa
import cv2
import httpx
import secrets
import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import yt_dlp
import jinja2

# ─── Auth config ─────────────────────────────────────────────────────────────
AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "viz")
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "chang3th1s!23")  # ← CHANGE THIS
_active_sessions: set[str] = set()


def check_auth(request: Request) -> bool:
    session_id = request.cookies.get("viz_sid")
    if session_id and session_id in _active_sessions:
        return True
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth_header.split(" ", 1)[1]).decode("utf-8")
            user, pwd = decoded.split(":", 1)
            return user == AUTH_USERNAME and pwd == AUTH_PASSWORD
        except Exception:
            pass
    return False


async def auth_middleware(request: Request, call_next):
    if (request.url.path.startswith("/api/health")
            or request.url.path.startswith("/static")):
        return await call_next(request)
    if check_auth(request):
        return await call_next(request)
    if request.url.path in ("/", "", "/index.html"):
        return HTMLResponse(
            '<html><body style="display:flex;align-items:center;justify-content:center;'
            'height:100vh;font-family:sans-serif;background:#08060f;color:#a78bfa">'
            '<div style="text-align:center;width:100%;max-width:400px">'
            '<h1 style="font-size:2rem;margin-bottom:.5rem">🔒 Locked</h1>'
            '<p style="color:#7a72aa;margin-bottom:1.5rem">Enter credentials to access the visualizer</p>'
            '<form method="POST" action="/api/login" style="display:flex;flex-direction:column;gap:.6rem">'
            '<input name="username" placeholder="Username" required autofocus '
            'style="padding:.6rem;border-radius:8px;border:1px solid #2a2548;'
            'background:#141126;color:#c9c2e6;font-size:.9rem">'
            '<input name="password" type="password" placeholder="Password" required '
            'style="padding:.6rem;border-radius:8px;border:1px solid #2a2548;'
            'background:#141126;color:#c9c2e6;font-size:.9rem">'
            '<button type="submit" style="padding:.6rem;border-radius:8px;'
            'border:none;background:#7c5cfc;color:#fff;font-weight:600;'
            'font-size:.9rem;cursor:pointer">Login</button>'
            '</form>'
            '</div></body></html>',
            status_code=401,
        )
    return HTMLResponse(
        '<html><body style="display:flex;align-items:center;justify-content:center;'
        'height:100vh;font-family:sans-serif;background:#08060f;color:#f06d6d">'
        '<div style="text-align:center">'
        '<h1 style="font-size:1.5rem;margin-bottom:.5rem">✗ Unauthorized</h1>'
        '<p style="color:#7a72aa;margin-bottom:1.5rem">You do not have access.</p>'
        '<a href="/" style="color:#a78bfa;text-decoration:none;'
        'border:1px solid #2a2548;padding:.5rem 1rem;border-radius:8px;'
        'display:inline-block">Go to login</a>'
        '</div></body></html>',
        status_code=401,
    )


# ─── Imports ────────────────────────────────────────────────────────────────

from backend.analysis.audio_analysis import AudioAnalyzer
from backend.visuals.visual_engine import VisualEngine, infer_theme, THEME_PRESETS
from backend.visuals.ai_keyframes import AIKeyframeInjector, DemoImageProvider, Keyframe
from backend.video.video_assembler import VideoAssembler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Visualizer AI", version="1.0")
app.middleware("http")(auth_middleware)

# ─── Directories ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(BASE_DIR / "templates")),
    cache_size=0,
)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class VisualRequest(BaseModel):
    url: str
    theme: str = "default"
    resolution: str = "1080p"
    fps: int = 60


class ProcessingStatus(BaseModel):
    status: str
    progress: float = 0.0
    message: str = ""
    output_path: Optional[str] = None
    _ai_prompts: Optional[list] = None


# ─── Job store ───────────────────────────────────────────────────────────────

jobs: dict[str, ProcessingStatus] = {}


# ─── Download helper ─────────────────────────────────────────────────────────

async def download_audio(url: str, output_path: Path) -> Path:
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav",
                             "preferredquality": "192"}],
        "outtmpl": str(output_path / "%(id)s.%(ext)s"),
        "quiet": True, "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)
        for f in output_path.iterdir():
            if f.suffix == ".wav":
                return f
        raise RuntimeError(f"No audio file found. URL: {url}")


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return HTMLResponse(content=_jinja_env.get_template("index.html").render(request=request))


@app.post("/api/process")
async def process_video(url: str = Form(...), theme: str = Form("default"),
                        resolution: str = Form("1080p"), fps: int = Form(60)):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = ProcessingStatus(status="queued", message="Queued...", progress=0.0)
    asyncio.create_task(run_job(job_id, url, theme, resolution, fps))
    return {"id": job_id, **jobs[job_id].model_dump()}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/api/download/{job_id}")
async def download_video(job_id: str):
    if job_id not in jobs or not jobs[job_id].output_path:
        raise HTTPException(404, "Not ready")
    path = Path(jobs[job_id].output_path)
    if not path.exists():
        raise HTTPException(404, "File missing")
    return StreamingResponse(open(path, "rb"), media_type="video/mp4",
                             headers={"Content-Disposition": f"attachment; filename={path.name}"})


@app.get("/api/themes")
async def list_themes():
    return {"themes": ["cosmic-space", "neon-cyberpunk", "ocean-deep", "firestorm",
                       "geometric-minimal", "photo-realistic", "anime-style", "vaporwave",
                       "nature-forest", "abstract-particles", "dark-gothic", "retro-80s"]}


@app.get("/api/keyframes/prompts/{job_id}")
async def get_ai_prompts(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    prompts = getattr(jobs[job_id], '_ai_prompts', None)
    return {"prompts": prompts or [], "count": len(prompts or []),
            "message": "No AI keyframes generated yet" if not prompts else ""}


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0"}


# ─── Auth endpoints ──────────────────────────────────────────────────────────

@app.post("/api/login")
async def login(request: Request):
    form = await request.form()
    user, pwd = form.get("username", ""), form.get("password", "")
    if user == AUTH_USERNAME and pwd == AUTH_PASSWORD:
        sid = secrets.token_hex(16)
        _active_sessions.add(sid)
        resp = HTMLResponse(
            '<html><script>window.location.href = window.location.origin'
            '+ window.location.pathname</script><body><p>✓ Logged in. Redirecting...</p></body></html>')
        resp.set_cookie("viz_sid", sid, max_age=86400 * 30, httponly=True, samesite="lax")
        return resp
    return HTMLResponse(
        '<html><body style="display:flex;align-items:center;justify-content:center;'
        'height:100vh;font-family:sans-serif;background:#08060f;color:#f06d6d">'
        '<div style="text-align:center">'
        '<h1 style="font-size:1.5rem;margin-bottom:.5rem">✗ Wrong Credentials</h1>'
        '<p style="color:#7a72aa;margin-bottom:1.5rem">Try again.</p>'
        '<form method="GET" action="/">'
        '<button type="submit" style="padding:.6rem 1.2rem;border-radius:8px;'
        'border:1px solid #2a2548;background:#141126;color:#c9c2e6;'
        'font-size:.9rem;cursor:pointer">Try Again</button>'
        '</form></div></body></html>',
        status_code=401,
    )


@app.get("/api/logout")
async def logout(request: Request):
    resp = RedirectResponse(url="/")
    resp.delete_cookie("viz_sid")
    return resp


# ─── Background job runner ───────────────────────────────────────────────────

async def run_job(job_id: str, url: str, theme: str, resolution: str, fps: int):
    job = jobs[job_id]
    try:
        job.status, job.message, job.progress = "downloading", "Downloading audio...", 5
        audio_path = await download_audio(url, UPLOAD_DIR)
        job.progress, job.message = 10, f"Downloaded: {audio_path.name}"

        job.status, job.message = "analyzing", "Analyzing beats, tempo, energy..."
        analyzer = AudioAnalyzer(str(audio_path))
        analysis = analyzer.analyze()
        duration, beats = analysis["duration"], analysis["beats"]
        tempo, energy = analysis["tempo"], analysis["energy"]
        spectral_flux = analysis.get("spectral_flux", [])
        job.progress, job.message = 30, f"Found {len(beats)} beats @ {tempo:.1f} BPM, {duration:.1f}s"

        res_map = {"1080p": (1920, 1080), "1440p": (2560, 1440), "4K": (3840, 2160)}
        width, height = res_map.get(resolution, (1920, 1080))
        engine = VisualEngine(width, height, theme=theme)
        theme_config = engine.theme

        job.status, job.message = "generating", "Generating AI keyframes for beat drops..."
        kf_dir = UPLOAD_DIR / f"{job_id}_keyframes"
        kf_dir.mkdir(parents=True, exist_ok=True)
        injector = AIKeyframeInjector(theme=theme_config, keyframes_dir=kf_dir,
                                       max_keyframes=8, image_provider=DemoImageProvider(kf_dir),
                                       theme_name=theme if theme in THEME_PRESETS else None)
        beat_drops = engine._detect_beat_drops(beats, energy, spectral_flux)  # type: ignore
        ai_keyframes = injector.inject(beat_drops)
        num_ai = len([kf for kf in ai_keyframes if kf.generated])
        job.message = f"AI keyframes ready: {num_ai}/{len(ai_keyframes)} images loaded"
        job.progress = 45
        job._ai_prompts = injector.get_prompts_for_review()

        job.status, job.message = "generating", "Generating visuals with AI keyframes..."
        frames = engine.render(beats=beats, energy=energy, spectral_flux=spectral_flux,
                                tempo=tempo, fps=fps, duration=duration, ai_keyframes=ai_keyframes)
        job.progress, job.message = 75, f"Generated {len(frames)} frames (with {len(ai_keyframes)} AI keyframes)"

        job.status, job.message = "encoding", "Encoding final video with FFmpeg..."
        assembler = VideoAssembler(str(audio_path), str(OUTPUT_DIR / f"{job_id}.mp4"))
        output_path = assembler.assemble(frames, fps)
        job.progress, job.status, job.message = 100, "done", "Done! 🎉"
        job.output_path = output_path
        logger.info(f"Job {job_id} complete: {output_path}")

    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        job.status, job.message = "error", f"Error: {str(e)}"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
