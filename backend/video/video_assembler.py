"""
Video Assembler
Takes generated frames + the original audio and produces a final MP4 via FFmpeg.
"""

import subprocess
import tempfile
import shutil
from pathlib import Path

import numpy as np
import cv2
from PIL import Image


class VideoAssembler:
    """Takes generated frames + original audio and produces an MP4 via FFmpeg."""

    def __init__(self, audio_path: str, output_path: str):
        self.audio_path = audio_path
        self.output_path = output_path
        self._frame_dir = None

    def assemble(self, frames: list, fps: int = 60) -> str:
        """
        Write frames to a temp directory as PNGs, then use FFmpeg to combine
        with the audio into an MP4 (H.264 + AAC).
        Returns the output file path.
        """
        self._frame_dir = tempfile.mkdtemp(prefix="viz_frames_")
        frame_path = Path(self._frame_dir)

        for i, frame in enumerate(frames):
            out = frame_path / f"frame_{i:06d}.png"
            if isinstance(frame, np.ndarray):
                cv2.imwrite(str(out), frame)
            elif hasattr(frame, 'save'):   # PIL Image
                frame.save(str(out))
            else:
                raise TypeError(f"Unsupported frame type: {type(frame)}")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(frame_path / "frame_%06d.png"),
            "-i", self.audio_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            self.output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")

        shutil.rmtree(self._frame_dir, ignore_errors=True)
        self._frame_dir = None

        if not Path(self.output_path).exists():
            raise RuntimeError(f"Output file not created: {self.output_path}")
        return self.output_path

    def assemble_from_paths(self, frame_pattern: str, fps: int = 60) -> str:
        """Assemble using an existing frame image pattern (e.g. frames/frame_%06d.png)."""
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-i", self.audio_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            self.output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")
        return self.output_path
