"""
Audio Visualizer — Backend
FastAPI server with token-based authenticator.
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
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import yt_dlp
import jinja2

# ─── Username + Password authenticator ─────────────────────────────────────
# Set via env vars AUTH_USERNAME and AUTH_PASSWORD, or use these defaults.
# CHANGE THE DEFAULTS BELOW to something only you and your friend know.
AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "viz")
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "chang3th1s!23")  # ← CHANGE THIS

# Simple in-memory session store (use Redis in production)
_active_sessions: set[str] = set()


def check_auth(request: Request) -> bool:
    """Return True if request has a valid session cookie or Basic Auth."""
    # Check session cookie
    session_id = request.cookies.get("viz_sid")
    if session_id and session_id in _active_sessions:
        return True
    # Fallback: HTTP Basic Auth (for curl / direct access)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Basic "):
        import base64
        try:
            decoded = base64.b64decode(auth_header.split(" ", 1)[1]).decode("utf-8")
            user, pwd = decoded.split(":", 1)
            return user == AUTH_USERNAME and pwd == AUTH_PASSWORD
        except Exception:
            pass
    return False


async def auth_middleware(request: Request, call_next):
    """Middleware: require login for non-whitelisted routes."""
    # Whitelist: health check, static files
    if (request.url.path.startswith("/api/health")
            or request.url.path.startswith("/static")):
        return await call_next(request)

    if check_auth(request):
        return await call_next(request)

    # No credentials — show login page on /, error on other paths
    if request.url.path in ("/", "", "/index.html"):
        return HTMLResponse(
            '<html><body style="display:flex;align-items:center;justify-content:center;'
            'height:100vh;font-family:sans-serif;background:#08060f;color:#a78bfa">'
            '<div style="text-align:center;width:100%;max-width:400px">'
            '<h1 style="font-size:2rem;margin-bottom:.5rem">🔒 Locked</h1>'
            '<p style="color:#7a72aa;margin-bottom:1.5rem">Enter your credentials to access the visualizer</p>'
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
        '<p style="color:#7a72aa;margin-bottom:1.5rem">You do not have access to this page.</p>'
        '<a href="/" style="color:#a78bfa;text-decoration:none;'
        'border:1px solid #2a2548;padding:.5rem 1rem;border-radius:8px;'
        'display:inline-block">Go to login</a>'
        '</div></body></html>',
        status_code=401,
    )


# ─── Login endpoint ─────────────────────────────────────────────────────────

@app.post("/api/login")
async def login(request: Request):
    """Accept username/password, set session cookie on success."""
    form = await request.form()
    user = form.get("username", "")
    pwd = form.get("password", "")
    if user == AUTH_USERNAME and pwd == AUTH_PASSWORD:
        import secrets
        session_id = secrets.token_hex(16)
        _active_sessions.add(session_id)
        response = HTMLResponse(
            '<html><script>window.location.href = window.location.origin + window.location.pathname'
            '</script><body><p>✓ Logged in. Redirecting...</p></body></html>'
        )
        response.set_cookie(
            key="viz_sid",
            value=session_id,
            max_age=86400 * 30,  # 30 days
            httponly=True,
            samesite="lax",
        )
        return response
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
        '</form>'
        '</div></body></html>',
        status_code=401,
    )


# ─── Logout endpoint ────────────────────────────────────────────────────────

@app.get("/api/logout")
async def logout(request: Request):
    """Clear the session cookie."""
    response = RedirectResponse(url="/")
    response.delete_cookie("viz_sid")
    return response
        '<html><body style="display:flex;align-items:center;justify-content:center;'
        'height:100vh;font-family:sans-serif;background:#08060f;color:#f06d6d">'
        '<div style="text-align:center">'
        '<h1 style="font-size:1.5rem;margin-bottom:.5rem">✗ Unauthorized</h1>'
        '<p style="color:#7a72aa;margin-bottom:1.5rem">Invalid or missing token.</p>'
        '<a href="/" style="color:#a78bfa;text-decoration:none;'
        'border:1px solid #2a2548;padding:.5rem 1rem;border-radius:8px;'
        'display:inline-block">Go to token entry</a>'
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

# ─── Directories ────────────────────────────────────────────────────────────

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


# ─── Job store ──────────────────────────────────────────────────────────────

jobs: dict[str, ProcessingStatus] = {}


# ─── Download helper ─────────────────────────────────────────────────────────

async def download_audio(url: str, output_path: Path) -> Path:
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "outtmpl": str(output_path / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)
        for f in output_path.iterdir():
            if f.suffix == ".wav":
                return f
        raise RuntimeError(f"No audio file found after download. URL: {url}")


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    template = _jinja_env.get_template("index.html")
    return HTMLResponse(content=template.render(request=request))


@app.post("/api/process")
async def process_video(
    url: str = Form(...),
    theme: str = Form("default"),
    resolution: str = Form("1080p"),
    fps: int = Form(60),
):
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
    return StreamingResponse(
        open(path, "rb"),
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename={path.name}"},
    )


@app.get("/api/themes")
async def list_themes():
    return {
        "themes": [
            "cosmic-space", "neon-cyberpunk", "ocean-deep", "firestorm",
            "geometric-minimal", "photo-realistic", "anime-style", "vaporwave",
            "nature-forest", "abstract-particles", "dark-gothic", "retro-80s",
        ]
    }


@app.get("/api/keyframes/prompts/{job_id}")
async def get_ai_prompts(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    prompts = getattr(jobs[job_id], '_ai_prompts', None)
    if prompts:
        return {"prompts": prompts, "count": len(prompts)}
    return {"prompts": [], "count": 0, "message": "No AI keyframes generated yet"}


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0"}


# ─── Background job runner ──────────────────────────────────────────────────

async def run_job(job_id: str, url: str, theme: str, resolution: str, fps: int):
    job = jobs[job_id]
    try:
        # 1. Download
        job.status = "downloading"
        job.message = "Downloading audio..."
        job.progress = 5
        audio_path = await download_audio(url, UPLOAD_DIR)
        job.progress = 10
        job.message = f"Downloaded: {audio_path.name}"

        # 2. Analyze
        job.status = "analyzing"
        job.message = "Analyzing beats, tempo, energy..."
        analyzer = AudioAnalyzer(str(audio_path))
        analysis = analyzer.analyze()
        duration = analysis["duration"]
        beats = analysis["beats"]
        tempo = analysis["tempo"]
        energy = analysis["energy"]
        spectral_flux = analysis.get("spectral_flux", [])
        job.progress = 30
        job.message = f"Found {len(beats)} beats @ {tempo:.1f} BPM, {duration:.1f}s duration"

        # 3. Create engine + resolve theme
        res_map = {"1080p": (1920, 1080), "1440p": (2560, 1440), "4K": (3840, 2160)}
        width, height = res_map.get(resolution, (1920, 1080))
        engine = VisualEngine(width, height, theme=theme)
        theme_config = engine.theme

        # 4. AI keyframe generation
        job.status = "generating"
        job.message = "Generating AI keyframes for beat drops..."
        keyframes_dir = UPLOAD_DIR / f"{job_id}_keyframes"
        keyframes_dir.mkdir(parents=True, exist_ok=True)
        theme_name = theme if theme in THEME_PRESETS else None
        injector = AIKeyframeInjector(
            theme=theme_config,
            keyframes_dir=keyframes_dir,
            max_keyframes=8,
            image_provider=DemoImageProvider(keyframes_dir),
            theme_name=theme_name,
        )
        beat_drops = engine._detect_beat_drops(beats, energy, spectral_flux)  # type: ignore
        ai_keyframes = injector.inject(beat_drops)
        num_ai = len([kf for kf in ai_keyframes if kf.generated])
        job.message = f"AI keyframes ready: {num_ai}/{len(ai_keyframes)} images loaded"
        job.progress = 45
        job._ai_prompts = injector.get_prompts_for_review()  # type: ignore[attr-defined]

        # 5. Render frames with AI keyframes composited
        job.status = "generating"
        job.message = "Generating visuals with AI keyframes composited..."
        frames = engine.render(
            beats=beats, energy=energy, spectral_flux=spectral_flux,
            tempo=tempo, fps=fps, duration=duration,
            ai_keyframes=ai_keyframes,
        )
        job.progress = 75
        job.message = f"Generated {len(frames)} frames (with {len(ai_keyframes)} AI keyframes)"

        # 6. Assemble video
        job.status = "encoding"
        job.message = "Encoding final video with FFmpeg..."
        assembler = VideoAssembler(str(audio_path), str(OUTPUT_DIR / f"{job_id}.mp4"))
        output_path = assembler.assemble(frames, fps)
        job.progress = 100
        job.status = "done"
        job.message = "Done! 🎉"
        job.output_path = output_path
        logger.info(f"Job {job_id} complete: {output_path}")

    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        job.status = "error"
        job.message = f"Error: {str(e)}"


# ─── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
