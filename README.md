# Audio Visualizer AI

Turn YouTube/SoundCloud links into beat-driven 1080p+ music visuals with AI keyframes.

## Quick start (local)

```bash
cd backend
pip install -r requirements.txt
VIZ_TOKEN=your-secret-token python main.py
```

Then open `http://localhost:8000/?token=your-secret-token`

## Deployment

### Render (recommended)

1. Push this repo to GitHub
2. In Render Dashboard → **New Web Service**
3. Connect your GitHub repo
4. Configure:
   - **Build Command:** `pip install -r requirements.txt` (or leave blank if using Dockerfile)
   - **Start Command:** `python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables:** `VIZ_TOKEN=your-secret-token`
   - **Plan:** Free tier works for low traffic
5. Deploy

Note: Render's free tier may spin down after inactivity. The audio-visualization processing (FFmpeg + OpenCV) is CPU-intensive; the free tier may be slow for full songs.

### Railway

1. Push this repo to GitHub
2. In Railway Dashboard → **New Project** → **Deploy from GitHub**
3. Configure:
   - **Environment Variables:** `VIZ_TOKEN=your-secret-token`
   - **Start Command:** `python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Railway automatically detects the Python app and installs dependencies

### Docker

```bash
docker build -t audio-viz .
docker run -p 8000:8000 -e VIZ_TOKEN=your-secret-token audio-viz
```

## Usage

1. Open `https://your-app/?token=your-secret-token`
2. Paste a YouTube or SoundCloud URL
3. Enter a visual theme (e.g. "cosmic space", "neon cyberpunk", "ocean deep")
4. Click **Generate Visuals**
5. Wait for processing — download the MP4 when done

## Environment Variables

| Variable | Description |
|---|---|
| `VIZ_TOKEN` | Shared secret token. Anyone with `?token=...` can access. |

## Tech stack

- **Backend:** FastAPI + uvicorn (Python)
- **Audio analysis:** librosa (beat detection, tempo, energy, spectral bands)
- **Visual engine:** OpenCV + NumPy (procedural particles, shapes, AI keyframe compositing)
- **Video assembly:** FFmpeg (H.264 + AAC MP4)
- **Audio download:** yt-dlp (YouTube/SoundCloud)
- **Frontend:** Vanilla HTML/CSS/JS (no build step)
