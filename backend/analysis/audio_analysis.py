"""
Audio Analysis Module
Uses librosa to extract beats, tempo, energy envelopes, and spectral features
from an audio file. All data is normalized and structured for the visual engine.
"""

import numpy as np
import librosa
import librosa.display
from pathlib import Path
import json


class AudioAnalyzer:
    """
    Analyze an audio file and extract visual-reactive features:
      - global tempo (BPM)
      - beat timestamps
      - onset strength (beat drops / hits)
      - energy envelope (RMS)
      - spectral flux (timbre changes)
      - spectral centroid (brightness)
      - low/mid/high frequency bands (for bass/midtro/high-reactive visuals)
    """

    def __init__(self, audio_path: str, hop_length: int = 512, n_fft: int = 2048):
        self.audio_path = audio_path
        self.hop_length = hop_length
        self.n_fft = n_fft
        self._y = None
        self._sr = None

    def load(self):
        if self._y is None:
            self._y, self._sr = librosa.load(self.audio_path, sr=None)
        return self._y, self._sr

    def analyze(self) -> dict:
        y, sr = self.load()
        duration = len(y) / sr

        # ── Tempo & beats ──────────────────────────────────────────────
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length)

        # ── Onset strength (beat drops) ────────────────────────────────
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr,
                                                   hop_length=self.hop_length, backtrack=True)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
        onset_strengths = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
        # onset strength values at detection points
        onset_vals = onset_env[onset_frames] if len(onset_frames) < len(onset_env) else onset_env

        # ── Energy (RMS) envelope ──────────────────────────────────────
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=self.hop_length)

        # ── Spectral features ──────────────────────────────────────────
        D = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length))
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft,
                                             hop_length=self.hop_length)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                              n_fft=self.n_fft,
                                                              hop_length=self.hop_length)[0]
        spectral_flux = librosa.onset.onset_strength(y=y, sr=sr,
                                                      hop_length=self.hop_length)
        # freq band energies (normalized 0-1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        band_energies = {}
        for name, (lo, hi) in {
            "low": (20, 250),       # kick / bass
            "mid": (250, 2000),    # vocals / lead
            "high": (2000, 16000), # air / hats
        }.items():
            mask = (freqs >= lo) & (freqs <= hi)
            band_energies[name] = np.mean(D[mask], axis=0) if mask.any() else np.zeros(D.shape[1])

        # ── Normalize helpers ──────────────────────────────────────────
        def normalize(arr):
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-9:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        return {
            "sample_rate": int(sr),
            "duration": float(duration),
            "tempo": float(np.atleast_1d(tempo)[0]),
            "beats": beat_times.tolist(),
            "beat_frames": beat_frames.tolist(),
            "onsets": onset_times.tolist(),
            "onset_strengths": onset_vals.tolist() if hasattr(onset_vals, 'tolist') else [],
            "energy": normalize(rms).tolist(),
            "energy_times": rms_times.tolist(),
            "spectral_centroid": normalize(spectral_centroid).tolist(),
            "spectral_flux": normalize(spectral_flux).tolist(),
            "low_band": normalize(band_energies["low"]).tolist(),
            "mid_band": normalize(band_energies["mid"]).tolist(),
            "high_band": normalize(band_energies["high"]).tolist(),
            "mfcc": [],  # placeholder — can be expanded
        }

    def save_analysis(self, out_path: str):
        data = self.analyze()
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        return data

    def get_beat_drop_indices(self, threshold: float = 0.7) -> list[int]:
        """Return frame indices where onset strength exceeds threshold."""
        _, sr = self.load()
        analysis = self.analyze()
        strengths = np.array(analysis["onset_strengths"])
        if len(strengths) == 0:
            return []
        idxs = np.where(strengths > threshold)[0]
        return idxs.tolist()

    def get_tempo_changes(self, window: float = 8.0) -> list[dict]:
        """Detect sections where tempo shifts significantly."""
        y, sr = self.load()
        # Use a sliding window tempo analysis
        win_len = int(window * sr)
        step = int(0.5 * sr)
        segments = []
        t = 0.0
        while t + window < y.shape[0]:
            seg = y[int(t * sr):int((t + window) * sr)]
            if len(seg) > sr:
                tempo, _ = librosa.beat.beat_track(y=seg, sr=sr)
                segments.append({"time": t, "tempo": float(tempo[0])})
            t += step * 0.5
        # Find jumps
        changes = []
        for i in range(1, len(segments)):
            prev = segments[i - 1]["tempo"]
            curr = segments[i]["tempo"]
            if abs(curr - prev) / max(prev, 1) > 0.15:
                changes.append({
                    "time": segments[i]["time"],
                    "old_tempo": prev,
                    "new_tempo": curr,
                    "delta": curr - prev,
                })
        return changes
