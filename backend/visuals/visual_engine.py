"""
Visual Engine
Generates 1080p+ reactive frames based on audio analysis data.
Supports theming via text prompts — colors, particle behavior, shapes, and motion style
are derived from the theme string using keyword-based presets + procedural generation.

The engine produces individual PNG frames that are later assembled into an MP4 by FFmpeg.
"""

import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import colorsys
import random
import math
from typing import List, Dict, Optional
from dataclasses import dataclass, field

# Try to import AI keyframe types (optional dependency)
try:
    from .ai_keyframes import Keyframe
except ImportError:
    Keyframe = None  # type: ignore


# ─── Theme presets ──────────────────────────────────────────────────────────

THEME_PRESETS = {
    "cosmic-space": {
        "bg": (5, 5, 20),
        "accent": (100, 180, 255),
        "secondary": (180, 100, 255),
        "particle_color": (150, 200, 255),
        "particle_type": "glow_dot",
        "shape": "circle",
        "motion": "float_slow",
        "intensity": 1.0,
    },
    "neon-cyberpunk": {
        "bg": (5, 0, 15),
        "accent": (0, 255, 200),
        "secondary": (255, 0, 100),
        "particle_color": (255, 100, 255),
        "particle_type": "neon_line",
        "shape": "triangle",
        "motion": "pulse_fast",
        "intensity": 1.3,
    },
    "ocean-deep": {
        "bg": (0, 10, 30),
        "accent": (0, 100, 200),
        "secondary": (0, 200, 180),
        "particle_color": (0, 200, 255),
        "particle_type": "bubble",
        "shape": "wave",
        "motion": "drift",
        "intensity": 0.9,
    },
    "firestorm": {
        "bg": (15, 2, 0),
        "accent": (255, 80, 0),
        "secondary": (255, 200, 50),
        "particle_color": (255, 150, 50),
        "particle_type": "spark",
        "shape": "flame",
        "motion": "rise_fast",
        "intensity": 1.4,
    },
    "geometric-minimal": {
        "bg": (20, 20, 30),
        "accent": (255, 255, 255),
        "secondary": (180, 180, 255),
        "particle_color": (220, 220, 255),
        "particle_type": "line",
        "shape": "polygon",
        "motion": "rotate",
        "intensity": 0.7,
    },
    "vaporwave": {
        "bg": (20, 0, 30),
        "accent": (255, 100, 200),
        "secondary": (100, 200, 255),
        "particle_color": (255, 150, 220),
        "particle_type": "grid_dot",
        "shape": "square",
        "motion": "glitch",
        "intensity": 1.1,
    },
    "nature-forest": {
        "bg": (5, 20, 5),
        "accent": (50, 200, 80),
        "secondary": (100, 255, 120),
        "particle_color": (80, 220, 100),
        "particle_type": "leaf",
        "shape": "organic",
        "motion": "sway",
        "intensity": 0.8,
    },
    "dark-gothic": {
        "bg": (0, 0, 5),
        "accent": (80, 0, 80),
        "secondary": (180, 0, 80),
        "particle_color": (150, 0, 150),
        "particle_type": "smoke",
        "shape": "crystal",
        "motion": "drift_slow",
        "intensity": 0.9,
    },
    "retro-80s": {
        "bg": (10, 0, 20),
        "accent": (255, 150, 50),
        "secondary": (100, 200, 255),
        "particle_color": (255, 200, 100),
        "particle_type": "star",
        "shape": "star",
        "motion": "roll",
        "intensity": 1.0,
    },
    "abstract-particles": {
        "bg": (10, 5, 15),
        "accent": (255, 100, 100),
        "secondary": (100, 200, 255),
        "particle_color": (255, 150, 200),
        "particle_type": "blob",
        "shape": "blob",
        "motion": "chaos",
        "intensity": 1.2,
    },
    "anime-style": {
        "bg": (20, 20, 40),
        "accent": (255, 120, 200),
        "secondary": (100, 200, 255),
        "particle_color": (255, 180, 220),
        "particle_type": "sparkle",
        "shape": "ring",
        "motion": "burst",
        "intensity": 1.2,
    },
    "photo-realistic": {
        "bg": (20, 20, 25),
        "accent": (255, 255, 255),
        "secondary": (200, 200, 200),
        "particle_color": (255, 255, 255),
        "particle_type": "light_ray",
        "shape": "ray",
        "motion": "sweep",
        "intensity": 0.6,
    },
}


def infer_theme(theme_str: str) -> dict:
    """Match a free-text theme to the closest preset; fall back to a generated palette."""
    ts = theme_str.lower().strip()
    if ts in THEME_PRESETS:
        return THEME_PRESETS[ts]

    # keyword matching
    keywords = {
        "space": "cosmic-space", "cosmic": "cosmic-space", "star": "cosmic-space",
        "cyberpunk": "neon-cyberpunk", "neon": "neon-cyberpunk", "tech": "neon-cyberpunk",
        "ocean": "ocean-deep", "sea": "ocean-deep", "water": "ocean-deep", "deep": "ocean-deep",
        "fire": "firestorm", "flame": "firestorm", "burn": "firestorm", "hot": "firestorm",
        "minimal": "geometric-minimal", "geometric": "geometric-minimal",
        "vaporwave": "vaporwave", "retro": "retro-80s", "80s": "retro-80s",
        "forest": "nature-forest", "nature": "nature-forest", "green": "nature-forest",
        "gothic": "dark-gothic", "dark": "dark-gothic", "goth": "dark-gothic",
        "anime": "anime-style", "jp": "anime-style", "japanese": "anime-style",
        "abstract": "abstract-particles", "particle": "abstract-particles",
    }
    for kw, preset in keywords.items():
        if kw in ts:
            return THEME_PRESETS[preset]

    # Fallback: generate a palette from the text
    return generate_fallback_theme(theme_str)


def generate_fallback_theme(theme_str: str) -> dict:
    """Create a theme by hashing the string into colors and picking a motion style."""
    h = hash(theme_str) % (2 ** 31)
    r1, g1, b1 = (h % 256), ((h >> 8) % 256), ((h >> 16) % 256)
    r2 = (r1 + 60) % 256
    g2 = (g1 + 60) % 256
    b2 = (b1 + 60) % 256
    bg = (max(r1 - 30, 0), max(g1 - 30, 0), max(b1 - 30, 0))
    return {
        "bg": bg,
        "accent": (r1, g1, b1),
        "secondary": (r2, g2, b2),
        "particle_color": (min(r1 + 40, 255), min(g1 + 40, 255), min(b1 + 40, 255)),
        "particle_type": "glow_dot",
        "shape": "circle",
        "motion": "float_slow",
        "intensity": 1.0,
        "_custom": True,
        "_prompt": theme_str,
    }


# ─── Particle system ─────────────────────────────────────────────────────────

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    size: float
    life: float  # 0..1
    type: str     # 'dot', 'glow', 'line', 'star', etc.
    color: tuple  # RGB
    rotation: float = 0.0
    rot_speed: float = 0.0
    phase: float = 0.0


class VisualEngine:
    """
    Renders reactive frames given audio analysis data and a theme.
    Outputs a list of numpy arrays (H, W, 3) in BGR (OpenCV) format.
    """

    def __init__(self, width: int = 1920, height: int = 1080, theme: str = "default"):
        self.width = width
        self.height = height
        self.theme = infer_theme(theme)
        self.particles: List[Particle] = []
        self.frame_idx = 0
        self.time = 0.0
        self.beats: List[float] = []
        self.energy: List[float] = []
        self.spectral_flux: List[float] = []
        self.beat_drop_queue: List[float] = []  # times of upcoming beat drops
        self._init_particles(count=400)
        # AI keyframes
        self.ai_keyframes: List[Keyframe] = []  # type: ignore
        self._loaded_keyframe_images: Dict[str, np.ndarray] = {}

    # ── Particle init ──────────────────────────────────────────────────────

    def _init_particles(self, count: int = 400):
        px, py = self.width, self.height
        ptype = self.theme["particle_type"]
        col = self.theme["particle_color"]
        for _ in range(count):
            x = random.uniform(0, px)
            y = random.uniform(0, py)
            vx = random.uniform(-20, 20)
            vy = random.uniform(-20, 20)
            size = random.uniform(2, 8)
            life = random.uniform(0.3, 1.0)
            rot = random.uniform(0, math.pi * 2)
            rot_speed = random.uniform(-0.02, 0.02)
            phase = random.uniform(0, math.pi * 2)
            self.particles.append(Particle(x, y, vx, vy, size, life, ptype, col, rot, rot_speed, phase))

    def _reset_particles(self):
        self._init_particles(count=400)

    # ── Rendering ──────────────────────────────────────────────────────────

    def render(
        self,
        beats: List[float],
        energy: List[float],
        spectral_flux: List[float] = None,
        tempo: float = 120.0,
        fps: int = 60,
        duration: float = 60.0,
        ai_keyframes: Optional[List[Keyframe]] = None,  # type: ignore
    ) -> List[np.ndarray]:
        """
        Generate all frames for the given audio.
        beats: list of beat timestamps (seconds)
        energy: normalized energy per frame (0-1)
        spectral_flux: timbre-change energy
        tempo: BPM
        fps: output frame rate
        duration: total seconds
        """
        self.beats = beats
        self.energy = energy
        self.spectral_flux = spectral_flux or []
        self.ai_keyframes = ai_keyframes or []
        # Pre-load AI keyframe images
        self._preload_keyframe_images()
        total_frames = int(fps * duration)
        frames = []

        # build beat-drop timeline
        beat_drops = self._detect_beat_drops(beats, energy, spectral_flux)

        # Pre-render: create frame buffer once
        for i in range(total_frames):
            t = i / fps
            self.time = t
            self.frame_idx = i
            frame = self._render_frame(t, i, fps, beat_drops)
            frames.append(frame)

            if i % 300 == 0:
                pass  # could emit progress callback

        return frames

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _star_polygon(cx: float, cy: float, outer_r: float, inner_r: float, points: int = 5) -> np.ndarray:
        """Create a star polygon (numpy array of int points)."""
        pts = []
        for i in range(points * 2):
            a = i * math.pi / points - math.pi / 2
            r = outer_r if i % 2 == 0 else inner_r
            pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
        return np.array(pts, np.int32)

    def _detect_beat_drops(self, beats, energy, spectral_flux) -> List[dict]:
        """Identify beat-drop moments (onset peaks) from the data."""
        drops = []
        # Use energy peaks as beat drop candidates
        e_arr = np.array(energy) if energy else np.zeros(1)
        threshold = 0.6
        for idx in range(1, len(e_arr) - 1):
            if e_arr[idx] > threshold and e_arr[idx] > e_arr[idx - 1] and e_arr[idx] > e_arr[idx + 1]:
                t = idx / 60.0  # assuming 60fps energy data
                drops.append({"time": t, "intensity": float(e_arr[idx]), "index": idx})
        # Also include detected beat times if available
        for b in beats:
            # Boost existing beat times if no energy data
            if not drops or abs(b - drops[-1]["time"]) > 0.1:
                drops.append({"time": b, "intensity": 0.7, "index": int(b * 60)})
        return drops

    def _render_frame(self, t: float, frame_idx: int, fps: int, beat_drops: List[dict]) -> np.ndarray:
        """Render a single frame at time t."""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        bg = self.theme["bg"]
        canvas[:, :] = bg

        # ── 1. Background gradient / atmosphere ──────────────────────────
        self._draw_atmosphere(canvas, t)

        # ── 2. Beat-reactive energy waves ────────────────────────────────
        energy_val = self._get_energy_at(t)
        self._draw_energy_waves(canvas, t, energy_val, beat_drops)

        # ── 3. Particles ──────────────────────────────────────────────────
        self._update_particles(t, energy_val, beat_drops)
        self._draw_particles(canvas, t)

        # ── 4. Central reactive element ───────────────────────────────────
        self._draw_central_element(canvas, t, energy_val, beat_drops)

        # ── 5. Beat drop bursts ───────────────────────────────────────────
        self._draw_beat_bursts(canvas, t, beat_drops)

        # ── 5b. AI keyframe overlay ────────────────────────────────────────
        canvas = self._composite_ai_keyframe(canvas, t, beat_drops)

        # ── 6. Vignette ───────────────────────────────────────────────────
        self._apply_vignette(canvas)

        return canvas

    # ── Sub-renderers ──────────────────────────────────────────────────────

    def _get_energy_at(self, t: float) -> float:
        if not self.energy:
            return 0.5
        idx = min(int(t * 60), len(self.energy) - 1)
        return float(self.energy[idx]) if idx >= 0 else 0.5

    def _get_spectral_at(self, t: float) -> float:
        if not self.spectral_flux:
            return 0.0
        idx = min(int(t * 60), len(self.spectral_flux) - 1)
        return float(self.spectral_flux[idx])

    def _draw_atmosphere(self, canvas: np.ndarray, t: float):
        """Slow-moving ambient background effects — color drift, aurora."""
        accent = np.array(self.theme["accent"], dtype=np.float32)
        bg = np.array(self.theme["bg"], dtype=np.float32)
        intensity = self.theme.get("intensity", 1.0)
        # subtle color pulsing in background
        pulse = math.sin(t * 0.5) * 0.05 * intensity
        drift = np.array([math.sin(t * 0.1) * 10, math.cos(t * 0.13) * 10, 0])
        # Add a subtle glow band
        for y in range(self.height):
            v = (y / self.height) * 0.15 * intensity
            color = bg + (accent - bg) * v * (0.5 + 0.5 * math.sin(t * 0.3 + y * 0.003))
            canvas[y, :] = np.clip(color, 0, 255).astype(np.uint8)

    def _draw_energy_waves(self, canvas: np.ndarray, t: float, energy: float, drops: List[dict]):
        """Horizontal waveform bands that react to energy."""
        accent = self.theme["accent"]
        wave_height = int(10 + energy * 40 * self.theme.get("intensity", 1.0))
        if wave_height < 2:
            return
        y_center = self.height - wave_height - 20
        for x in range(0, self.width, 4):
            phase = x * 0.02 + t * 4.0
            wave_val = math.sin(phase) * 0.5 + 0.5
            wave_val *= energy
            alpha = int(wave_val * 180 * self.theme.get("intensity", 1.0))
            if alpha > 0:
                color = (
                    min(accent[0] + int(wave_val * 60), 255),
                    min(accent[1] + int(wave_val * 40), 255),
                    min(accent[2] + int(wave_val * 80), 255),
                )
                canvas[y_center:y_center + wave_height, x:x + 4] = color

    def _update_particles(self, t: float, energy: float, drops: List[dict]):
        """Update particle positions and properties each frame."""
        px, py = self.width, self.height
        motion = self.theme.get("motion", "float_slow")
        intensity = self.theme.get("intensity", 1.0)
        speed_mult = 1.0 + energy * 2.0 * intensity

        # beat drop burst: push particles outward
        burst_active = False
        burst_strength = 0.0
        for d in drops:
            dt = t - d["time"]
            if 0 < dt < 0.5:
                burst_active = True
                burst_strength = d["intensity"] * (1 - dt / 0.5) * 3.0

        for p in self.particles:
            # motion style
            if motion == "pulse_fast":
                p.vx += math.sin(t * 8 + p.phase) * 0.3 * speed_mult
                p.vy += math.cos(t * 8 + p.phase) * 0.3 * speed_mult
            elif motion == "float_slow":
                p.vx += math.sin(t * 0.3 + p.phase) * 0.1
                p.vy += math.cos(t * 0.4 + p.phase) * 0.1
            elif motion == "rise_fast":
                p.vy -= 0.5 * speed_mult
                p.vx += math.sin(t + p.phase) * 0.2
            elif motion == "drift":
                p.vx = math.sin(t * 0.1 + p.phase) * 0.5
                p.vy = math.cos(t * 0.1 + p.phase) * 0.3
            elif motion == "rotate":
                p.rotation += p.rot_speed * speed_mult
                px_c = px / 2
                py_c = py / 2
                dx = p.x - px_c
                dy = p.y - py_c
                p.x = px_c + dx * math.cos(0.001 * speed_mult) - dy * math.sin(0.001 * speed_mult)
                p.y = py_c + dx * math.sin(0.001 * speed_mult) + dy * math.cos(0.001 * speed_mult)
                continue
            elif motion == "sway":
                p.vx = math.sin(t * 0.5 + p.phase) * 0.8
                p.vy = math.cos(t * 0.3 + p.phase * 2) * 0.3
            elif motion == "chaos":
                p.vx += random.uniform(-1, 1) * 0.3
                p.vy += random.uniform(-1, 1) * 0.3
            elif motion == "glitch":
                if random.random() < 0.02:
                    p.x += random.uniform(-50, 50)
                    p.y += random.uniform(-50, 50)
            elif motion == "burst":
                p.vx += math.sin(t * 3 + p.phase) * 0.5
                p.vy += math.cos(t * 3 + p.phase) * 0.5
            elif motion == "roll":
                p.rotation += 0.01 * speed_mult
            elif motion == "sweep":
                p.vx += math.sin(t * 0.2) * 0.2
            elif motion == "drift_slow":
                p.vx += math.sin(t * 0.05 + p.phase) * 0.05
                p.vy += math.cos(t * 0.07 + p.phase) * 0.05
            else:
                p.vx += random.uniform(-0.1, 0.1)
                p.vy += random.uniform(-0.1, 0.1)

            # burst
            if burst_active:
                dx = p.x - px / 2
                dy = p.y - py / 2
                dist = math.sqrt(dx * dx + dy * dy) + 1
                push = burst_strength / dist
                p.vx += (dx / dist) * push
                p.vy += (dy / dist) * push

            # apply velocity
            p.x += p.vx * speed_mult * 0.05
            p.y += p.vy * speed_mult * 0.05

            # damping
            p.vx *= 0.98
            p.vy *= 0.98

            # wrap around
            if p.x < -10: p.x = px + 10
            if p.x > px + 10: p.x = -10
            if p.y < -10: p.y = py + 10
            if p.y > py + 10: p.y = -10

            # life decay / regen
            p.life -= 0.002
            if p.life <= 0:
                p.x = random.uniform(0, px)
                p.y = random.uniform(0, py)
                p.life = random.uniform(0.3, 1.0)
                p.size = random.uniform(2, 8)

    def _draw_particles(self, canvas: np.ndarray, t: float):
        """Draw all particles to the canvas."""
        for p in self.particles:
            if p.life <= 0:
                continue
            alpha = min(p.life * 255, 255)
            size = max(p.size, 1)
            col = p.color
            x, y = int(p.x), int(p.y)

            ptype = p.type
            if ptype in ("glow_dot", "dot", "bubble", "sparkle"):
                # glow circle
                if 0 <= x < self.width and 0 <= y < self.height:
                    cv2.circle(canvas, (x, y), int(size), col, -1)
                    if size > 3:
                        cv2.circle(canvas, (x, y), int(size * 2), col, 1)
            elif ptype == "neon_line":
                cv2.circle(canvas, (x, y), int(size), col, 1)
            elif ptype == "star":
                if 0 <= x < self.width and 0 <= y < self.height:
                    cv2.drawMarker(canvas, (x, y), col, cv2.MARKER_STAR, int(size * 2), 1)
            elif ptype in ("line", "grid_dot"):
                cv2.circle(canvas, (x, y), 1, col, -1)
            elif ptype == "flame":
                if size > 2:
                    cv2.circle(canvas, (x, y), int(size), (255, 150, 50), -1)
            elif ptype == "smoke":
                cv2.circle(canvas, (x, y), int(size * 2), (180, 180, 180), -1)
            elif ptype == "leaf":
                pts = np.array([
                    [x, y - size],
                    [x + size * 0.5, y],
                    [x, y + size],
                    [x - size * 0.5, y],
                ], np.int32)
                cv2.fillPoly(canvas, [pts], col)
            elif ptype in ("crystal", "polygon", "organic", "blob"):
                cv2.circle(canvas, (x, y), int(size), col, -1)
            elif ptype == "light_ray":
                cv2.line(canvas,
                         (x - int(size), y - int(size)),
                         (x + int(size), y + int(size)),
                         col, 1)
            elif ptype == "ring":
                cv2.circle(canvas, (x, y), int(size), col, 2)
            else:
                cv2.circle(canvas, (x, y), int(size), col, -1)

    def _draw_central_element(self, canvas: np.ndarray, t: float, energy: float, drops: List[dict]):
        """Central visual that anchors the frame — pulsing shape, beat-reactive."""
        shape = self.theme.get("shape", "circle")
        accent = self.theme["accent"]
        secondary = self.theme.get("secondary", accent)
        width, height = self.width, self.height
        cx, cy = width // 2, height // 2
        energy_val = energy * self.theme.get("intensity", 1.0)

        base_size = int(60 + energy_val * 100)
        if base_size < 10:
            base_size = 10

        if shape == "circle":
            cv2.circle(canvas, (cx, cy), base_size, accent, -1)
            cv2.circle(canvas, (cx, cy), base_size + 5, secondary, 2)
            cv2.circle(canvas, (cx, cy), base_size + 12, accent, 1)
        elif shape == "triangle":
            pts = np.array([
                [cx, cy - base_size],
                [cx - base_size, cy + base_size * 0.6],
                [cx + base_size, cy + base_size * 0.6],
            ], np.int32)
            cv2.fillPoly(canvas, [pts], accent)
            cv2.polylines(canvas, [pts], True, secondary, 2)
        elif shape == "square":
            pt = (cx - base_size, cy - base_size, cx + base_size, cy + base_size)
            cv2.rectangle(canvas, (pt[0], pt[1]), (pt[2], pt[3]), accent, -1)
            cv2.rectangle(canvas, (pt[0], pt[1]), (pt[2], pt[3]), secondary, 2)
        elif shape == "star":
            pts = self._star_polygon(cx, cy, base_size, base_size // 2, 5)
            cv2.fillPoly(canvas, [pts], accent)
            pts2 = self._star_polygon(cx, cy, base_size + 8, base_size // 2, 5)
            cv2.polylines(canvas, [pts2], True, secondary, 2)
        elif shape == "polygon":
            pts = []
            sides = 6
            for i in range(sides):
                a = i * 2 * math.pi / sides - math.pi / 2 + t * 0.2
                pts.append([cx + base_size * math.cos(a), cy + base_size * math.sin(a)])
            pts = np.array(pts, np.int32)
            cv2.fillPoly(canvas, [pts], accent)
            cv2.polylines(canvas, [pts], True, secondary, 2)
        elif shape == "wave":
            # horizontal wave band through center
            for x in range(0, width, 2):
                phase = x * 0.03 + t * 3.0
                y_off = int(math.sin(phase) * energy_val * 40)
                yy = cy + y_off
                if 0 <= yy < height and yy + 4 < height:
                    cv2.line(canvas, (x, yy), (x, yy + 4), accent, 2)
        elif shape == "flame":
            # flame shape at center
            for layer in range(5):
                r = base_size - layer * 8
                if r < 5:
                    break
                color = (
                    min(255, accent[0] + layer * 30),
                    min(200, accent[1] - layer * 20),
                    max(0, accent[2] - layer * 30),
                )
                cv2.circle(canvas, (cx, cy + layer * 3), r, color, -1)
        elif shape == "organic":
            pts = []
            n = 12
            for i in range(n):
                a = i * 2 * math.pi / n
                r = base_size * (0.7 + 0.3 * math.sin(a * 3 + t * 2))
                pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
            pts = np.array(pts, np.int32)
            cv2.fillPoly(canvas, [pts], accent)
            cv2.polylines(canvas, [pts], True, secondary, 2)
        elif shape == "ray":
            # light rays from center
            for i in range(12):
                a = i * 2 * math.pi / 12 + t * 0.1
                ex = cx + math.cos(a) * base_size * 2
                ey = cy + math.sin(a) * base_size * 2
                cv2.line(canvas, (cx, cy), (int(ex), int(ey)), accent, max(1, int(energy_val * 4)))
        elif shape == "blob":
            pts = []
            n = 10
            for i in range(n):
                a = i * 2 * math.pi / n
                r = base_size * (0.8 + 0.2 * math.sin(a * 2 + t * 1.5))
                pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
            pts = np.array(pts, np.int32)
            cv2.fillPoly(canvas, [pts], accent)
        else:
            cv2.circle(canvas, (cx, cy), base_size, accent, -1)

    def _draw_beat_bursts(self, canvas: np.ndarray, t: float, drops: List[dict]):
        """Explosive burst at beat drops."""
        for d in drops:
            dt = t - d["time"]
            if 0 <= dt < 0.4:
                intensity = d["intensity"] * (1 - dt / 0.4)
                radius = int(20 + intensity * 80 * self.theme.get("intensity", 1.0))
                alpha = int(intensity * 80)
                accent = self.theme["accent"]
                secondary = self.theme.get("secondary", accent)

                # expanding ring
                cv2.circle(canvas, (self.width // 2, self.height // 2), radius,
                           secondary, 2)
                # flash
                if alpha > 5:
                    overlay = canvas.copy()
                    cv2.circle(overlay, (self.width // 2, self.height // 2), radius,
                               accent, -1)
                    canvas = cv2.addWeighted(canvas, 1.0, overlay, min(alpha / 255, 0.4), 0)

                # rays
                for i in range(8):
                    a = i * math.pi / 4 + dt * 5
                    ex = self.width // 2 + math.cos(a) * radius * 1.5
                    ey = self.height // 2 + math.sin(a) * radius * 1.5
                    cv2.line(canvas,
                             (self.width // 2, self.height // 2),
                             (int(ex), int(ey)),
                             accent, max(1, int(intensity * 3)))

        return canvas

    def _apply_vignette(self, canvas: np.ndarray):
        """Darken edges for cinematic feel."""
        h, w = canvas.shape[:2]
        vignette = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(vignette, (w // 2, h // 2), int(min(w, h) // 2 * 0.8), 255, -1)
        vignette = cv2.GaussianBlur(vignette, (51, 51), 0)
        vignette = 255 - vignette  # invert: dark at edges
        vignette = cv2.cvtColor(vignette, cv2.COLOR_GRAY2BGR)
        canvas = cv2.addWeighted(canvas, 1.0, vignette, 0.35, 0)
        return canvas

    # ── AI Keyframe Compositing ──────────────────────────────────────────────

    def _preload_keyframe_images(self):
        """Load all AI keyframe images into memory."""
        for kf in self.ai_keyframes:
            if kf.image_path and Path(kf.image_path).exists():
                try:
                    img = cv2.imread(kf.image_path)
                    if img is not None:
                        self._loaded_keyframe_images[kf.image_path] = img
                except Exception:
                    pass

    def _composite_ai_keyframe(self, canvas: np.ndarray, t: float,
                                beat_drops: List[dict]) -> np.ndarray:
        """
        At each AI keyframe time, composite the AI image with a dramatic
        entrance: flash + zoom-in, hold briefly, then crossfade back to procedural.

        Keyframe timing:
          - 0.0–0.15s before: flash + procedural elements intensify
          - 0.0s (keyframe time): AI image appears full-frame with zoom-in
          - 0.0–0.3s: hold AI image with subtle procedural overlay
          - 0.3–0.6s: crossfade back to procedural
        """
        for kf in self.ai_keyframes:
            dt = t - kf.time
            if -0.2 < dt < 0.6:
                intensity = kf.intensity
                img = self._loaded_keyframe_images.get(kf.image_path)
                if img is None:
                    continue

                # Resize AI image to canvas size
                ai_img = cv2.resize(img, (self.width, self.height))

                if -0.15 < dt <= 0.0:
                    # Pre-flash: procedural elements intensify, brief flash overlay
                    flash_alpha = (dt + 0.15) / 0.15  # 0..1 approaching keyframe
                    flash_overlay = np.zeros_like(canvas)
                    flash_overlay[:, :] = self.theme["accent"]
                    fade = flash_alpha * 0.2 * intensity
                    canvas = cv2.addWeighted(canvas, 1.0, flash_overlay, fade, 0)

                    # Scale AI image from center — zoom-in effect
                    scale = 0.3 + 0.7 * flash_alpha
                    new_w = int(self.width * scale)
                    new_h = int(self.height * scale)
                    scaled = cv2.resize(ai_img, (new_w, new_h))
                    x_off = (self.width - new_w) // 2
                    y_off = (self.height - new_h) // 2

                    overlay = np.zeros_like(canvas)
                    overlay[y_off:y_off + new_h, x_off:x_off + new_w] = scaled
                    alpha = 0.85 * flash_alpha * intensity
                    canvas = cv2.addWeighted(canvas, 1.0 - alpha, overlay, alpha, 0)

                elif 0.0 < dt <= 0.3:
                    # Hold: AI image full-frame with subtle procedural glow
                    hold_strength = 1.0 - (dt / 0.3) * 0.15  # slight fade over hold
                    overlay = np.zeros_like(canvas)
                    overlay[:, :] = ai_img
                    alpha = 0.92 * hold_strength
                    canvas = cv2.addWeighted(canvas, 1.0 - alpha, overlay, alpha, 0)

                    # Subtle particle glow on top
                    if intensity > 0.6:
                        glow = np.zeros_like(canvas)
                        glow[:, :] = self.theme["accent"]
                        glow_alpha = 0.05 * intensity * (1 - dt / 0.3)
                        canvas = cv2.addWeighted(canvas, 1.0, glow, glow_alpha, 0)

                elif 0.3 < dt < 0.6:
                    # Crossfade back to procedural
                    fade_out = 1.0 - (dt - 0.3) / 0.3  # 1..0
                    overlay = np.zeros_like(canvas)
                    overlay[:, :] = ai_img
                    alpha = 0.92 * fade_out
                    canvas = cv2.addWeighted(canvas, 1.0, overlay, 1.0 - alpha, 0)
                    # Actually blend the other way: procedural fading in
                    proc_weight = (dt - 0.3) / 0.3  # 0..1
                    # Re-do: AI fading out, procedural fading in
                    canvas2 = np.zeros_like(canvas)
                    canvas2[:, :] = ai_img
                    alpha_ai = 0.92 * (1 - proc_weight)
                    canvas = cv2.addWeighted(canvas, 1.0 - alpha_ai, canvas2, alpha_ai, 0)

        return canvas


# ─── standalone frame demo ──────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    engine = VisualEngine(1920, 1080, theme="neon-cyberpunk")
    frames = engine.render(
        beats=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        energy=[0.1, 0.2, 0.8, 0.3, 0.9, 0.4, 0.7, 0.5],
        spectral_flux=[0.1, 0.1, 0.9, 0.2, 0.95, 0.3, 0.8, 0.4],
        tempo=128,
        fps=60,
        duration=4.0,
    )
    print(f"Generated {len(frames)} frames")
    for i, f in enumerate(frames[:5]):
        cv2.imwrite(f"/home/user/audio-viz/demo_frame_{i}.png", f)
        print(f"  frame {i}: {f.shape}")
    print("Done — demo frames written.")
