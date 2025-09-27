from __future__ import annotations

import math
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .jobs import job_store
from .storage import get_result_path


# --- Utilities ---

def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", errors="ignore")


# --- Real pipeline steps ---

def extract_audio_from_video(video_path: Path, extracted_wav_path: Path) -> None:
    """Extract mono 16 kHz PCM WAV from video using ffmpeg."""
    extracted_wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(extracted_wav_path),
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        # save stderr to help debugging
        _write_text(extracted_wav_path.parent / "ffmpeg_extract.stderr.txt", proc.stderr.decode(errors="ignore"))
        raise RuntimeError("ffmpeg extract failed (see ffmpeg_extract.stderr.txt)")


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV file as mono float32 array in [-1, 1] and return (audio, sr)."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError("Expected 16-bit PCM WAV")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    return audio, sr


def _frame_signal(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if audio.ndim != 1:
        raise ValueError("audio must be mono")
    num_frames = 1 + max(0, (len(audio) - frame_length) // hop_length)
    if num_frames <= 0:
        return np.zeros((0, frame_length), dtype=np.float32)
    strides = (audio.strides[0] * hop_length, audio.strides[0])
    shape = (num_frames, frame_length)
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
    return np.array(frames, copy=True)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    f_min = 0.0
    f_max = sr / 2.0
    m_min = _hz_to_mel(np.array([f_min]))[0]
    m_max = _hz_to_mel(np.array([f_max]))[0]
    m_points = np.linspace(m_min, m_max, num=n_mels + 2)
    f_points = _mel_to_hz(m_points)
    bin_points = np.floor((n_fft // 2 + 1) * f_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_left = bin_points[m - 1]
        f_m_center = bin_points[m]
        f_m_right = bin_points[m + 1]
        if f_m_center == f_m_left:
            f_m_center += 1
        if f_m_right == f_m_center:
            f_m_right += 1
        # Rising slope
        for k in range(f_m_left, f_m_center):
            fb[m - 1, k] = (k - f_m_left) / max(1, (f_m_center - f_m_left))
        # Falling slope
        for k in range(f_m_center, f_m_right):
            fb[m - 1, k] = (f_m_right - k) / max(1, (f_m_right - f_m_center))
    # Normalize
    enorm = 2.0 / (f_points[2 : n_mels + 2] - f_points[:n_mels])
    for m in range(n_mels):
        fb[m] *= enorm[m]
    return fb


def _dct_matrix(n_mfcc: int, n_mels: int) -> np.ndarray:
    dct = np.zeros((n_mfcc, n_mels), dtype=np.float32)
    factor = math.pi / float(n_mels)
    for k in range(n_mfcc):
        for n in range(n_mels):
            dct[k, n] = math.cos((n + 0.5) * k * factor)
    dct[0, :] *= math.sqrt(1.0 / n_mels)
    for k in range(1, n_mfcc):
        dct[k, :] *= math.sqrt(2.0 / n_mels)
    return dct


def _mfcc(audio: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 512, n_mels: int = 40, n_mfcc: int = 13) -> tuple[np.ndarray, float]:
    frames = _frame_signal(audio, n_fft, hop_length)
    if frames.shape[0] == 0:
        return np.zeros((0, n_mfcc), dtype=np.float32), hop_length / float(sr)
    window = np.hanning(n_fft).astype(np.float32)
    frames = frames * window[None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (np.abs(spec) ** 2).astype(np.float32)
    fb = _mel_filterbank(sr, n_fft, n_mels)
    mel = np.maximum(1e-10, power @ fb.T)  # shape: [frames, n_mels]
    log_mel = np.log(mel).astype(np.float32)
    dct = _dct_matrix(n_mfcc, n_mels)
    mfcc = log_mel @ dct.T  # shape: [frames, n_mfcc]
    hop_seconds = hop_length / float(sr)
    return mfcc, hop_seconds


def _rms_envelope(audio: np.ndarray, sr: int, win_seconds: float = 0.05, hop_seconds: float = 0.01) -> tuple[np.ndarray, float]:
    """Compute RMS envelope downsampled for coarse cross-correlation.
    Returns (envelope, hop_s)
    """
    win = max(1, int(round(win_seconds * sr)))
    hop = max(1, int(round(hop_seconds * sr)))
    frames = _frame_signal(audio, win, hop)
    if frames.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32), hop / float(sr)
    rms = np.sqrt(np.mean(frames * frames, axis=1)).astype(np.float32)
    # Normalize
    if np.max(rms) > 0:
        rms = rms / np.max(rms)
    return rms, hop / float(sr)


def _xcorr_offset_seconds(env_ref: np.ndarray, env_tgt: np.ndarray, hop_s: float, max_lag_s: float = 30.0) -> float:
    """Estimate offset via cross-correlation of energy envelopes.
    Positive return value means target must be delayed.
    """
    if len(env_ref) == 0 or len(env_tgt) == 0:
        return 0.0
    # Mean-center
    a = env_ref - np.mean(env_ref)
    b = env_tgt - np.mean(env_tgt)
    # Normalize
    denom = (np.std(a) + 1e-8) * (np.std(b) + 1e-8)
    a /= (np.std(a) + 1e-8)
    b /= (np.std(b) + 1e-8)
    # Full cross-correlation
    xcorr = np.correlate(a, b, mode="full")
    lags = np.arange(-len(b) + 1, len(a))
    # Restrict to max lag window
    max_lag_frames = int(round(max_lag_s / hop_s))
    mask = (lags >= -max_lag_frames) & (lags <= max_lag_frames)
    xcorr = xcorr[mask]
    lags = lags[mask]
    if xcorr.size == 0:
        return 0.0
    best = int(lags[int(np.argmax(xcorr))])
    # If best > 0, target is behind ref (needs delay)
    return float(best) * hop_s


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.dot(d, d)))


def _dtw_offset(ref: np.ndarray, tgt: np.ndarray, hop_seconds: float, max_offset_s: float = 5.0) -> float:
    """Constrained DTW to estimate global offset (target relative to reference).
    Returns positive if target needs a delay, negative if target should advance.
    """
    n_ref = ref.shape[0]
    n_tgt = tgt.shape[0]
    if n_ref == 0 or n_tgt == 0:
        return 0.0
    band = int(max(1, round(max_offset_s / hop_seconds)))
    inf = 1e18
    cost = np.full((n_ref + 1, n_tgt + 1), inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n_ref + 1):
        j_min = max(1, i - band)
        j_max = min(n_tgt, i + band)
        for j in range(j_min, j_max + 1):
            d = _euclidean(ref[i - 1], tgt[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    # Backtrack to collect alignment path
    i, j = n_ref, n_tgt
    pairs: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        pairs.append((i - 1, j - 1))
        prev = min((cost[i - 1, j], 0), (cost[i, j - 1], 1), (cost[i - 1, j - 1], 2), key=lambda x: x[0])[1]
        if prev == 0:
            i -= 1
        elif prev == 1:
            j -= 1
        else:
            i -= 1
            j -= 1
    if not pairs:
        return 0.0
    deltas = np.array([j - i for (i, j) in pairs], dtype=np.float64)
    offset_frames = float(np.median(deltas))
    return offset_frames * hop_seconds


def estimate_offset(reference_audio_path: Path, target_audio_path: Path) -> float:
    """Estimate global offset between reference (extracted from video) and target clean audio.
    Always convert target to WAV mono 16 kHz 16-bit first for robustness.
    Positive return value means target must be delayed by that many seconds.
    """
    ref, sr_ref = _read_wav_mono(reference_audio_path)
    # Convert target to 16 kHz mono 16-bit WAV unconditionally (handles mp3/m4a/24-bit WAV, etc.)
    resampled = target_audio_path.parent / "clean_16k.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(target_audio_path),
        "-ac",
        "1",
        "-ar",
        str(sr_ref),
        "-acodec",
        "pcm_s16le",
        str(resampled),
    ]
    proc = _run(cmd)
    if proc.returncode != 0:
        _write_text(target_audio_path.parent / "ffmpeg_resample.stderr.txt", proc.stderr.decode(errors="ignore"))
        raise RuntimeError("ffmpeg resample failed (see ffmpeg_resample.stderr.txt)")
    tgt, sr_tgt = _read_wav_mono(resampled)
    # Coarse alignment via RMS envelope cross-correlation (±30s)
    env_ref, env_hop = _rms_envelope(ref, sr_ref, win_seconds=0.05, hop_seconds=0.01)
    env_tgt, _ = _rms_envelope(tgt, sr_tgt, win_seconds=0.05, hop_seconds=0.01)
    coarse = _xcorr_offset_seconds(env_ref, env_tgt, hop_s=env_hop, max_lag_s=30.0)
    # Prepare MFCC features
    ref = ref / (np.max(np.abs(ref)) + 1e-8)
    tgt = tgt / (np.max(np.abs(tgt)) + 1e-8)
    mfcc_ref, hop_s = _mfcc(ref, sr_ref)
    mfcc_tgt, _ = _mfcc(tgt, sr_tgt)
    # Mean-variance normalize along time
    if mfcc_ref.shape[0] > 0:
        mfcc_ref = (mfcc_ref - np.mean(mfcc_ref, axis=0)) / (np.std(mfcc_ref, axis=0) + 1e-8)
    if mfcc_tgt.shape[0] > 0:
        mfcc_tgt = (mfcc_tgt - np.mean(mfcc_tgt, axis=0)) / (np.std(mfcc_tgt, axis=0) + 1e-8)
    # Trim sequences according to coarse offset to center DTW
    shift_frames = int(round(coarse / hop_s)) if hop_s > 0 else 0
    ref_trim = mfcc_ref
    tgt_trim = mfcc_tgt
    if shift_frames > 0 and shift_frames < mfcc_tgt.shape[0]:
        tgt_trim = mfcc_tgt[shift_frames:]
    elif shift_frames < 0 and (-shift_frames) < mfcc_ref.shape[0]:
        ref_trim = mfcc_ref[-shift_frames:]
    # Equalize lengths for stability (optional)
    min_len = min(ref_trim.shape[0], tgt_trim.shape[0])
    if min_len <= 5:
        return float(coarse)
    ref_trim = ref_trim[:min_len]
    tgt_trim = tgt_trim[:min_len]
    residual = _dtw_offset(ref_trim, tgt_trim, hop_s, max_offset_s=2.0)
    return float(coarse + residual)


def replace_audio(video_path: Path, clean_audio_path: Path, output_path: Path, offset_seconds: float) -> None:
    """Mux original video with clean audio shifted by offset_seconds.
    If offset_seconds > 0: delay audio. If < 0: trim audio start by |offset|.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ms = int(round(abs(offset_seconds) * 1000.0))
    if offset_seconds >= 0:
        # Delay clean audio
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(clean_audio_path),
            "-filter_complex",
            f"[1:a]adelay={ms}:all=1[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-shortest",
            str(output_path),
        ]
    else:
        # Advance audio by trimming the start
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(abs(offset_seconds)),
            "-i",
            str(clean_audio_path),
            "-i",
            str(video_path),
            "-map",
            "1:v:0",
            "-map",
            "0:a:0",
            "-c:v",
            "copy",
            "-shortest",
            str(output_path),
        ]
    proc = _run(cmd)
    if proc.returncode != 0:
        # save stderr to job results dir
        _write_text(output_path.parent / "ffmpeg_mux.stderr.txt", proc.stderr.decode(errors="ignore"))
        raise RuntimeError("ffmpeg mux failed (see ffmpeg_mux.stderr.txt)")


def transcribe_fr(audio_path: Path) -> str:
    return ""


def detect_cuts(transcript_text: str) -> Tuple[list[Tuple[float, float]], list[float]]:
    return ([], [])


def apply_cuts_and_zooms(video_path: Path, cuts: list[Tuple[float, float]], zoom_peaks: list[float], output_path: Path) -> None:
    shutil.copyfile(video_path, output_path)


def run_real_pipeline(job_id: str, video_path: Path, clean_audio_path: Path) -> Path:
    job_store.update(job_id, status="running", progress=5, message="Preparing")
    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg not found in PATH")

    out_path = get_result_path(job_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Extract video audio
    job_store.update(job_id, progress=15, message="Extracting video audio")
    extracted = out_path.parent / "extracted.wav"
    extract_audio_from_video(video_path, extracted)

    # 2) Estimate offset via MFCC + constrained DTW
    job_store.update(job_id, progress=45, message="Estimating offset (MFCC/DTW)")
    try:
        offset = estimate_offset(extracted, clean_audio_path)
    except Exception as exc:
        job_store.update(job_id, status="failed", progress=100, message=f"Offset estimation failed: {exc}")
        raise
    # persist the offset for inspection
    try:
        _write_text(out_path.parent / "offset_seconds.txt", f"{offset:.6f}\n")
    except Exception:
        pass

    # 3) Replace/mux audio with offset
    job_store.update(job_id, progress=75, message=f"Replacing audio (offset {offset:+.3f}s)")
    try:
        replace_audio(video_path, clean_audio_path, out_path, offset_seconds=offset)
    except Exception as exc:
        job_store.update(job_id, status="failed", progress=100, message=f"Audio replace failed: {exc}")
        raise

    job_store.update(job_id, status="completed", progress=100, message="Done", result_path=str(out_path))
    return out_path

