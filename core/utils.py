"""Utility functions for whisper_transcribe."""

import json
import os
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Tuple, Union

import numpy as np

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False


def validate_file_path(file_path: Union[str, Path], must_exist: bool = False) -> Path:
    """
    Validate and sanitize file path to prevent directory traversal attacks.

    Args:
        file_path: Path to validate
        must_exist: If True, raises error if file doesn't exist

    Returns:
        Validated absolute Path object

    Raises:
        ValueError: If path contains suspicious patterns or doesn't exist (when must_exist=True)
    """
    try:
        path = Path(file_path).resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid file path: {e}")

    # Check for suspicious patterns
    path_str = str(path)
    if '..' in Path(file_path).parts:
        raise ValueError(f"Path traversal detected in: {file_path}")

    if must_exist and not path.exists():
        raise ValueError(f"File does not exist: {file_path}")

    return path


def format_timestamp(seconds: float) -> str:
    """
    Format seconds to HH:MM:SS timestamp format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string (HH:MM:SS)
    """
    if seconds is None:
        return "00:00:00"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable duration string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def load_audio_segment(
    audio_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    log: Optional[Callable[[str], None]] = None
) -> Tuple[np.ndarray, float]:
    """
    Load audio file as mono 16kHz float32, with optional segment selection.

    Shared by every transcription engine (HF transformers, faster-whisper):
    audio decoding is engine-agnostic, so this lives here instead of being
    duplicated per engine class.

    Args:
        audio_path: Path to audio file
        start_time: Start time in seconds (None = from beginning)
        end_time: End time in seconds (None = to end)
        log: Optional logging callback, e.g. self.log from a transcriber

    Returns:
        Tuple of (audio_array, duration_in_seconds)
    """
    def _log(message: str):
        if log:
            log(message)

    _log(f"Loading audio: {audio_path}")

    audio = None
    original_duration = 0

    # Try pydub first for better M4A/AAC support
    if HAS_PYDUB:
        try:
            _log("Trying pydub for audio loading...")
            audio_segment = AudioSegment.from_file(audio_path)
            original_duration = float(len(audio_segment) / 1000.0)  # milliseconds to seconds

            # Apply segment selection if specified
            if start_time is not None or end_time is not None:
                start_ms = int(start_time * 1000) if start_time is not None else 0
                end_ms = int(end_time * 1000) if end_time is not None else len(audio_segment)

                _log(f"Extracting segment: {start_time or 0:.2f}s - {end_time or original_duration:.2f}s")
                audio_segment = audio_segment[start_ms:end_ms]

            # Convert to mono 16kHz
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

            # Normalize based on sample width
            if audio_segment.sample_width == 2:
                audio = audio / 32768.0
            elif audio_segment.sample_width == 4:
                audio = audio / 2147483648.0

            duration = len(audio) / 16000.0
            _log(f"Successfully loaded audio with pydub. Duration: {duration:.2f}s")

        except Exception as e:
            _log(f"Pydub failed: {e}")
            audio = None

    # Fallback to librosa if pydub fails
    if audio is None:
        try:
            # Use soundfile backend to avoid audioread deprecation
            import soundfile as sf
            audio_data, sample_rate = sf.read(audio_path)

            # Calculate original duration
            original_duration = len(audio_data) / sample_rate

            # Apply segment selection if specified
            if start_time is not None or end_time is not None:
                start_sample = int(start_time * sample_rate) if start_time is not None else 0
                end_sample = int(end_time * sample_rate) if end_time is not None else len(audio_data)

                _log(f"Extracting segment: {start_time or 0:.2f}s - {end_time or original_duration:.2f}s")
                audio_data = audio_data[start_sample:end_sample]

            # Resample if needed
            import librosa
            if sample_rate != 16000:
                audio = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            else:
                audio = audio_data

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            duration = len(audio) / 16000.0
            _log(f"Successfully loaded audio with soundfile. Duration: {duration:.2f}s")

        except Exception as e:
            _log(f"Soundfile failed: {e}")
            # Final fallback: librosa (may show deprecation warnings)
            try:
                import librosa
                audio, sample_rate = librosa.load(audio_path, sr=16000)

                # For librosa fallback, we need to handle segment selection differently
                if start_time is not None or end_time is not None:
                    _log("Warning: Segment selection with librosa fallback may be less accurate")
                    start_sample = int(start_time * 16000) if start_time is not None else 0
                    end_sample = int(end_time * 16000) if end_time is not None else len(audio)
                    audio = audio[start_sample:end_sample]

                duration = len(audio) / 16000.0
                _log(f"Successfully loaded audio with librosa (with warnings). Duration: {duration:.2f}s")

            except Exception as e2:
                raise Exception(
                    f"All audio loading methods failed. "
                    f"Pydub: {e if HAS_PYDUB else 'Not available'}, "
                    f"Soundfile: {e}, Librosa: {e2}"
                )

    if audio is None or len(audio) == 0:
        raise Exception("Audio file appears to be empty or corrupted")

    duration = len(audio) / 16000.0
    return audio, duration


def save_transcript(
    result: Dict[str, Any],
    audio_path: str,
    output_dir: str,
    output_format: str = 'markdown',
    log: Optional[Callable[[str], None]] = None
) -> str:
    """
    Save a transcription result (as returned by any engine's transcribe_audio)
    to a Markdown or JSON file. Format-only; engine-agnostic.

    Args:
        result: Transcription result dict with keys: text, chunks, duration,
                processing_time
        audio_path: Original audio file path
        output_dir: Output directory
        output_format: 'markdown' or 'json'
        log: Optional logging callback

    Returns:
        Path to saved file
    """
    def _log(message: str):
        if log:
            log(message)

    audio_name = Path(audio_path).stem

    if output_format == 'json':
        output_file = Path(output_dir) / f"{audio_name}.json"
        _log(f"Saving transcript to JSON: {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'audio_file': Path(audio_path).name,
                'duration': result['duration'],
                'processing_time': result['processing_time'],
                'text': result['text'],
                'chunks': result['chunks']
            }, f, ensure_ascii=False, indent=2)

    else:  # markdown (default)
        output_file = Path(output_dir) / f"{audio_name}.md"
        _log(f"Saving transcript to Markdown: {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Transcript: {audio_name}\n\n")
            f.write(f"**Source:** {Path(audio_path).name}\n\n")
            f.write("## Content\n\n")
            f.write(result['text'])
            f.write("\n")

    _log("Transcript saved successfully")
    return str(output_file)
