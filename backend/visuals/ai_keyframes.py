"""
AI Keyframe Injector
Generates themed image prompts for major beat drops and manages AI-generated
keyframe images that get composited into the video at those moments.

Architecture:
  1. Beat drop detection → pick the most energetic drops (top N by intensity)
  2. For each drop, generate a rich image prompt based on the theme + drop context
  3. The prompts are sent to an image generation API (Stability AI, DALL-E, etc.)
  4. Generated images are stored and loaded by the VisualEngine at render time

This module is API-agnostic — it produces prompts + manages image files. The actual
generation is delegated to a provider. A demo provider using the conversation's
generate_image tool is included for local development.
"""

import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import math

# Import theme presets from visual engine
from .visual_engine import THEME_PRESETS


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class Keyframe:
    """One AI-generated image to be shown at a specific time."""
    time: float           # seconds into the song
    intensity: float      # 0..1, how strong this drop is
    prompt: str           # the image generation prompt
    image_path: Optional[str] = None   # path to generated image (after generation)
    generated: bool = False


# ─── Prompt generator ────────────────────────────────────────────────────────

class PromptGenerator:
    """
    Turns a theme + beat-drop context into a vivid image generation prompt.
    Prompts describe: a full-frame visual scene that matches the theme aesthetic,
    with energy/motion that reflects the beat drop strength.
    """

    SCENE_TEMPLATES = {
        "cosmic-space": [
            "A breathtaking deep space scene with swirling nebulae in {colors}, "
            "stars exploding outward from center, cosmic energy radiating in all directions, "
            "{intensity} lighting, cinematic, 8k resolution, hyperdetailed",
            "Deep space vortex with {colors} cosmic clouds, bright star burst at center, "
            "energy waves propagating through the nebula, dramatic {intensity} glow, "
            "photorealistic space photography, ultra wide angle",
            "Astral plane with {colors} ethereal light rays piercing through cosmic dust, "
            "a powerful energy surge at the center, {intensity} luminous particles, "
            "surreal cosmic artwork, rich detail",
        ],
        "neon-cyberpunk": [
            "Neon-lit cyberpunk city at night, {colors} neon signs reflecting on wet streets, "
            "a powerful energy surge lighting up the sky, holographic displays exploding outward, "
            "{intensity} vibrant neon palette, Blade Runner aesthetic, cinematic",
            "Futuristic digital cityscape, {colors} neon grid lines pulsing with energy, "
            "a massive data burst erupting from the center, {intensity} glowing cyber elements, "
            "high contrast, cyberpunk art",
            "Digital cyberspace realm, {colors} neon light streams converging and exploding, "
            "matrix-like energy cascade, {intensity} saturation, glitch art aesthetic, "
            "futuristic visualization",
        ],
        "ocean-deep": [
            "Deep ocean abyss with {colors} bioluminescent creatures emerging from the depths, "
            "powerful underwater energy waves rippling through the dark water, "
            "{intensity} glowing marine life, majestic and serene, National Geographic style",
            "Vast underwater scene, {colors} plankton illuminating the deep, "
            "a massive energy surge through the water column, {intensity} light rays from above, "
            "serene deep ocean atmosphere, photorealistic",
            "Deep sea trench with {colors} glowing jellyfish and luminous currents, "
            "powerful tidal energy waves, {intensity} ethereal underwater light, "
            "mysterious ocean depths, ultra detailed",
        ],
        "firestorm": [
            "Massive wildfire at night, {colors} flames erupting skyward with intense energy, "
            "embers flying in all directions, {intensity} heat distortion, "
            "dramatic firestorm visual, photorealistic, high contrast",
            "Infernal energy vortex of {colors} fire and embers, "
            "wildfire explosion with {intensity} intensity, flames licking upward, "
            "highly detailed fire particle system, cinematic lighting",
            "Volcanic eruption of {colors} fire and magma, "
            "massive energy explosion with {intensity} radiant heat, "
            "lava flows and ash clouds, dramatic natural disaster scene",
        ],
        "geometric-minimal": [
            "Minimalist geometric composition in {colors}, "
            "perfect geometric shapes expanding and fracturing with energy, "
            "{intensity} clean lines, Bauhaus aesthetic, abstract art, white background",
            "Abstract geometric art with {colors} clean shapes in motion, "
            "a burst of geometric fragmentation at center, {intensity} precision, "
            "modern art gallery aesthetic, symmetrical composition",
            "Geometric light installation, {colors} clean luminous polygons, "
            "expanding geometric pattern with {intensity} clarity, "
            "minimalist architectural visualization, sharp and modern",
        ],
        "vaporwave": [
            "Vaporwave dreamscape with {colors} pastel gradients, "
            "retro grid landscape with a powerful energy wave sweeping through, "
            "{intensity} pink and blue synthwave palette, nostalgic cyber aesthetic, "
            "1980s digital art",
            "Synthwave sunset with {colors} retro gradients, "
            "a massive digital energy wave transforming the landscape, "
            "{intensity} nostalgic 80s atmosphere, vaporwave art style, "
            "dreamy and nostalgic",
            "Digital retro landscape with {colors} checkerboard patterns and palm silhouettes, "
            "explosive energy wave of {intensity} pastel light, "
            "vaporwave aesthetic, surreal nostalgia",
        ],
        "nature-forest": [
            "Lush forest at twilight with {colors} magical light filtering through trees, "
            "a powerful energy surge through the woodland, {intensity} bioluminescent flora, "
            "enchanting nature scene, fantasy photography",
            "Ancient forest with {colors} glowing flora and energy streams flowing through, "
            "nature magic energy burst with {intensity} luminescence, "
            "fantasy forest atmosphere, ethereal and magical",
            "Misty forest clearing with {colors} dawn light rays, "
            "a surge of natural energy rippling through the trees, {intensity} soft glow, "
            "serene woodland scene, cinematic nature photography",
        ],
        "dark-gothic": [
            "Gothic cathedral interior with {colors} dark stained glass and mystical energy, "
            "powerful dark energy waves rippling through the space, {intensity} purple and crimson, "
            "dark fantasy atmosphere, dramatic shadows",
            "Dark crystalline cavern with {colors} glowing gothic arches, "
            "mystical energy surge with {intensity} crimson light, "
            "gothic horror aesthetic, moody and atmospheric",
            "Gothic night sky with {colors} ominous clouds and energy storms, "
            "dark mystical power surging across the sky, {intensity} dramatic darkness, "
            "gothic romanticism, atmospheric and dark",
        ],
        "retro-80s": [
            "1980s retro landscape with {colors} synthwave sunset and geometric grid, "
            "a powerful energy wave of {intensity} neon light transforming the scene, "
            "retro futurism, nostalgic and vibrant",
            "Retro arcade realm with {colors} glowing geometric elements, "
            "explosive energy burst of {intensity} 80s neon, "
            "classic arcade aesthetic, fun and nostalgic",
            "80s Memphis-style pattern explosion with {colors} bold geometric shapes, "
            "energy surge of {intensity} retro color, "
            "bold pop art aesthetic, playful and energetic",
        ],
        "abstract-particles": [
            "Abstract particle universe with {colors} glowing particles in chaos, "
            "massive particle explosion with {intensity} energy, "
            "constellation of luminous dots, generative art, vibrant",
            "Fluid abstract art with {colors} swirling energy patterns, "
            "powerful morphing energy burst with {intensity} color explosion, "
            "abstract expressionism, modern generative artwork",
            "Digital particle cosmos with {colors} organic blob shapes in motion, "
            "explosive generative energy with {intensity} chromatic variety, "
            "abstract digital art, flowing and chaotic",
        ],
        "anime-style": [
            "Anime-style magical energy explosion with {colors} sparkling light rays, "
            "powerful shonen-style energy burst with {intensity} radiant glow, "
            "anime keyframe aesthetic, dramatic and beautiful",
            "Japanese fantasy scene with {colors} magical circles and light spirits, "
            "intense energy surge with {intensity} sparkling particle effects, "
            "anime art style, cel-shaded, vibrant and magical",
            "Anime mecha cockpit view with {colors} dashboard and energy displays, "
            "massive power surge with {intensity} glowing effects, "
            "anime mechanical aesthetic, dynamic and intense",
        ],
        "photo-realistic": [
            "Cinematic light ray photography with {colors} volumetric god rays, "
            "powerful energy wave of {intensity} luminous light cutting through darkness, "
            "photorealistic, dramatic lighting, 8k resolution",
            "Realistic light beam installation with {colors} projected light patterns, "
            "explosive energy display with {intensity} brilliance, "
            "architectural photography, clean and cinematic",
            "Atmospheric light phenomenon with {colors} natural light bursting through clouds, "
            "{intensity} real-world lighting physics, "
            "cinematic photography, realistic and awe-inspiring",
        ],
    }

    def generate_prompts(self, theme: dict, beat_drops: List[dict], max_keyframes: int = 8,
                         theme_name: str = None) -> List[Keyframe]:
        """
        Generate Keyframe objects for the top beat drops.
        beat_drops: list of dicts with 'time', 'intensity', 'index'
        theme_name: optional explicit theme name (e.g. 'cosmic-space'). If None,
                    inferred from theme dict.
        """
        if not beat_drops:
            return []

        # Sort by intensity, take top N
        sorted_drops = sorted(beat_drops, key=lambda d: d["intensity"], reverse=True)
        selected = sorted_drops[:max_keyframes]
        selected.sort(key=lambda d: d["time"])

        # Determine which preset to use
        if theme_name:
            matched_preset = theme_name if theme_name in THEME_PRESETS else "abstract-particles"
        elif theme.get("_custom"):
            # Custom theme from text prompt — try keyword matching
            pt = (theme.get("_prompt", "") or "").lower()
            matched_preset = "abstract-particles"
            for pk in THEME_PRESETS.keys():
                if any(kw in pt for kw in pk.split("-")):
                    matched_preset = pk
                    break
        else:
            # Preset theme — match by comparing theme dict to presets
            matched_preset = "abstract-particles"
            theme_bg = tuple(theme.get("bg", (0, 0, 0)))
            for pk, pd in THEME_PRESETS.items():
                pd_bg = tuple(pd.get("bg", (0, 0, 0)))
                if theme_bg == pd_bg:
                    matched_preset = pk
                    break
            # Fallback: keyword match from accent color
            if matched_preset == "abstract-particles":
                accent = theme.get("accent", (0, 0, 0))
                # Very rough color-based matching
                if accent[2] > 180 and accent[0] < 100:  # blue accent
                    matched_preset = "cosmic-space"
                elif accent[0] > 200 and accent[1] < 100:  # red accent
                    matched_preset = "firestorm"

        templates = self.SCENE_TEMPLATES.get(matched_preset,
            self.SCENE_TEMPLATES["abstract-particles"])

        keyframes = []
        for i, drop in enumerate(selected):
            t = drop["time"]
            intensity = drop["intensity"]

            # Color description from theme
            accent = theme.get("accent", (100, 180, 255))
            secondary = theme.get("secondary", (180, 100, 255))
            colors = self._color_description(accent, secondary, theme)

            # Pick template, rotating through available ones
            tmpl = templates[i % len(templates)]
            prompt = tmpl.format(
                colors=colors,
                intensity=self._intensity_word(intensity),
            )

            # Add theme-specific detail
            prompt += self._theme_detail(theme, matched_preset, i)

            keyframes.append(Keyframe(
                time=t,
                intensity=intensity,
                prompt=prompt,
            ))

        return keyframes

    def _color_description(self, accent, secondary, theme) -> str:
        """Describe the theme colors in words for the prompt."""
        def name(r, g, b):
            if r > 200 and g < 80 and b < 80:  return "deep red"
            if r > 200 and g > 120 and b < 80: return "warm orange"
            if r > 200 and g > 180 and b < 100: return "golden yellow"
            if r < 80 and g > 150 and b < 80:  return "emerald green"
            if r < 80 and g < 80 and b > 200:  return "deep blue"
            if r < 100 and g > 150 and b > 150: return "cyan teal"
            if r > 130 and g < 80 and b > 150: return "rich purple"
            if r > 180 and g < 100 and b > 180: return "vibrant magenta"
            if r < 60 and g < 60 and b < 60:  return "near-black"
            if r > 220 and g > 220 and b > 220: return "pure white"
            if r > 180 and g > 180 and b > 180: return "soft pastel"
            # Blue-ish defaults (common in themes)
            if b > r and b > g: return "cool blue"
            if r > g and r > b: return "warm tones"
            if g > r and g > b: return "natural green"
            return "aurora"
        a_name = name(*accent)
        s_name = name(*secondary)
        if a_name == s_name:
            return f"{a_name} tones"
        return f"{a_name} and {s_name} accents"

    def _intensity_word(self, intensity: float) -> str:
        if intensity > 0.9: return "extremely powerful"
        if intensity > 0.7: return "powerful"
        if intensity > 0.5: return "strong"
        if intensity > 0.3: return "moderate"
        return "gentle"

    def _theme_detail(self, theme: dict, preset: str, idx: int) -> str:
        """Add extra prompt detail based on theme."""
        motion = theme.get("motion", "float_slow")
        shape = theme.get("shape", "circle")
        extra = ""
        if motion in ("pulse_fast", "burst", "chaos"):
            extra += " dynamic motion blur, high energy"
        elif motion in ("float_slow", "drift", "drift_slow"):
            extra += " serene and flowing, tranquil atmosphere"
        elif motion == "rotate":
            extra += " rotational symmetry, hypnotic motion"
        elif motion == "glitch":
            extra += " datamoshing effect, digital distortion"
        if shape == "star":
            extra += " radiant starburst elements"
        elif shape == "polygon":
            extra += " geometric crystalline structures"
        elif shape == "ray":
            extra += " dramatic light beams"
        return extra


# ─── Image provider interface ────────────────────────────────────────────────

class ImageProvider:
    """
    Abstract interface for generating images from prompts.
    Implement subclasses for different APIs (Stability, DALL-E, etc.)
    """

    def generate(self, prompt: str, output_path: Path, width: int = 1920, height: int = 1080) -> Path:
        raise NotImplementedError

    def generate_batch(self, keypoints: List[Keyframe], output_dir: Path,
                       width: int = 1920, height: int = 1080) -> List[Path]:
        paths = []
        for i, kf in enumerate(keyframes):
            out = output_dir / f"keyframe_{i:03d}_{kf.time:.2f}s.png"
            path = self.generate(kf.prompt, out, width, height)
            kf.image_path = str(path)
            kf.generated = True
            paths.append(path)
        return paths


# ─── Demo provider (uses generate_image tool via file convention) ────────────
# NOTE: In production, replace with StabilityAIProvider, DALL-E Provider, etc.
# This demo class documents the interface and stores prompts for manual review.


class DemoImageProvider(ImageProvider):
    """
    Demo provider that generates prompt files for review.
    In the live chat environment, use the `generate_image` tool separately
    and place the resulting images in the keyframes directory.
    """

    def __init__(self, keyframes_dir: Path):
        self.keyframes_dir = keyframes_dir
        self.keyframes_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, prompt: str, output_path: Path, width: int = 1920, height: int = 1080) -> Path:
        """
        In demo mode, this writes the prompt to a .prompt file.
        The user (or calling agent) should use generate_image with this prompt
        and save the result to output_path.
        """
        prompt_file = output_path.with_suffix(".prompt")
        prompt_file.write_text(f"WIDTH: {width}\nHEIGHT: {height}\n\n{prompt}\n")
        # If an image already exists at output_path, return it
        if output_path.exists():
            return output_path
        raise FileNotFoundError(
            f"Demo: No image found at {output_path}. "
            f"Prompt written to {prompt_file}. "
            f"Use generate_image with the prompt, save to {output_path}."
        )

    def generate_all_from_prompts(self, prompts: List[str], output_dir: Path) -> List[str]:
        """
        Write all prompts to files and return list of existing image paths.
        Caller should generate images for any that are missing.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for i, prompt in enumerate(prompts):
            out = output_dir / f"ai_keyframe_{i:03d}.png"
            self.generate(prompt, out)
            if out.exists():
                results.append(str(out))
            else:
                results.append(None)  # need to generate
        return results


# ─── Keyframe injector (orchestrator) ────────────────────────────────────────

class AIKeyframeInjector:
    """
    Orchestrates AI keyframe generation for a job:
      1. Takes audio analysis + theme
      2. Generates prompts for top beat drops
      3. (Optionally) calls image provider to generate images
      4. Returns Keyframe list for the visual engine to use
    """

    def __init__(self, theme: dict, keyframes_dir: Path,
                 max_keyframes: int = 8, image_provider: ImageProvider = None,
                 theme_name: str = None):
        self.theme = theme
        self.keyframes_dir = keyframes_dir
        self.max_keyframes = max_keyframes
        self.provider = image_provider or DemoImageProvider(keyframes_dir)
        self.theme_name = theme_name
        self.prompts: List[str] = []
        self.keyframes: List[Keyframe] = []

    def inject(self, beat_drops: List[dict]) -> List[Keyframe]:
        """Generate prompts and optionally generate images."""
        generator = PromptGenerator()
        self.keyframes = generator.generate_prompts(
            self.theme, beat_drops, self.max_keyframes,
            theme_name=self.theme_name,
        )
        self.prompts = [kf.prompt for kf in self.keyframes]

        # Try to generate images if provider supports it
        if hasattr(self.provider, 'generate_batch'):
            try:
                paths = self.provider.generate_batch(
                    self.keyframes,
                    self.keyframes_dir,
                )
            except Exception as e:
                # Demo mode: prompts written, images need manual generation
                pass

        # Load any already-existing images
        for kf in self.keyframes:
            if kf.image_path and Path(kf.image_path).exists():
                kf.generated = True

        return self.keyframes

    def get_prompts_for_review(self) -> List[dict]:
        """Return prompts for UI review / manual generation."""
        return [
            {"index": i, "time": kf.time, "intensity": kf.intensity, "prompt": kf.prompt}
            for i, kf in enumerate(self.keyframes)
        ]
