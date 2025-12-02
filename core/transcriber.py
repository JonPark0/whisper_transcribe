"""Core transcription module with API and progress tracking support."""

import time
import json
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import numpy as np

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

from .utils import format_timestamp


class WhisperTranscriber:
    """
    WhisperTranscriber provides audio transcription using OpenAI's Whisper model.

    This class can be used as a library (with progress callbacks) or through CLI.
    """

    def __init__(
        self,
        verbose: bool = False,
        chunk_length: int = 30,
        use_flash_attn: bool = False,
        target_language: Optional[str] = None
    ):
        """
        Initialize WhisperTranscriber.

        Args:
            verbose: Enable detailed logging
            chunk_length: Length of audio chunks in seconds (default: 30)
            use_flash_attn: Enable Flash Attention 2 for faster GPU processing
            target_language: Target language for translation (ISO 639-1 code)
        """
        self.verbose = verbose
        self.chunk_length = chunk_length
        self.use_flash_attn = use_flash_attn
        self.target_language = target_language
        self.model = None
        self.processor = None
        self.pipe = None
        self.device = None
        self.model_dtype = None

    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[INFO] {message}")

    def load_model(self):
        """Load Whisper large-v3-turbo model with appropriate device configuration."""
        self.log("Loading Whisper large-v3 turbo model...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3-turbo"

        # Prepare model loading arguments
        model_kwargs = {
            "dtype": dtype,
            "low_cpu_mem_usage": True,
            "use_safetensors": True
        }

        # Add flash attention support if requested and available
        if self.use_flash_attn and torch.cuda.is_available():
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                self.log("Flash Attention 2 enabled")
            except Exception as e:
                self.log(f"Flash Attention 2 not available, falling back to standard attention: {e}")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, **model_kwargs)
        self.model.to(device)

        # Store model dtype for input conversion
        self.model_dtype = dtype
        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_id)

        # Prepare generation config to fix deprecation warnings
        generation_config = {
            "max_new_tokens": 440,
            "return_timestamps": True,
        }

        # Add language settings if specified
        if self.target_language:
            generation_config["language"] = self.target_language
            generation_config["task"] = "translate"
            self.log(f"Translation enabled: translating to {self.target_language}")

        # Re-create pipeline for long-form transcription
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=self.chunk_length,
            batch_size=16,
            return_timestamps=True,
            device=device,
            generate_kwargs=generation_config
        )
        self.log(f"Model loaded successfully on {device} with chunk length {self.chunk_length}s")

    def load_audio_segment(
        self,
        audio_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Load audio file with optional segment selection.

        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds (None = from beginning)
            end_time: End time in seconds (None = to end)

        Returns:
            Tuple of (audio_array, duration_in_seconds)
        """
        self.log(f"Loading audio: {audio_path}")

        audio = None
        original_duration = 0

        # Try pydub first for better M4A/AAC support
        if HAS_PYDUB:
            try:
                self.log("Trying pydub for audio loading...")
                audio_segment = AudioSegment.from_file(audio_path)
                original_duration = len(audio_segment) / 1000.0  # milliseconds to seconds

                # Apply segment selection if specified
                if start_time is not None or end_time is not None:
                    start_ms = int(start_time * 1000) if start_time is not None else 0
                    end_ms = int(end_time * 1000) if end_time is not None else len(audio_segment)

                    self.log(f"Extracting segment: {start_time or 0:.2f}s - {end_time or original_duration:.2f}s")
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
                self.log(f"Successfully loaded audio with pydub. Duration: {duration:.2f}s")

            except Exception as e:
                self.log(f"Pydub failed: {e}")
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

                    self.log(f"Extracting segment: {start_time or 0:.2f}s - {end_time or original_duration:.2f}s")
                    audio_data = audio_data[start_sample:end_sample]

                # Resample if needed
                if sample_rate != 16000:
                    audio = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                else:
                    audio = audio_data

                # Convert to mono if stereo
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)

                duration = len(audio) / 16000.0
                self.log(f"Successfully loaded audio with soundfile. Duration: {duration:.2f}s")

            except Exception as e:
                self.log(f"Soundfile failed: {e}")
                # Final fallback: librosa (may show deprecation warnings)
                try:
                    audio, sample_rate = librosa.load(audio_path, sr=16000)

                    # For librosa fallback, we need to handle segment selection differently
                    if start_time is not None or end_time is not None:
                        self.log("Warning: Segment selection with librosa fallback may be less accurate")
                        start_sample = int(start_time * 16000) if start_time is not None else 0
                        end_sample = int(end_time * 16000) if end_time is not None else len(audio)
                        audio = audio[start_sample:end_sample]

                    duration = len(audio) / 16000.0
                    self.log(f"Successfully loaded audio with librosa (with warnings). Duration: {duration:.2f}s")

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

    def transcribe_audio(
        self,
        audio_path: str,
        enable_timestamps: bool = False,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio file with optional progress tracking.

        Args:
            audio_path: Path to audio file
            enable_timestamps: Include timestamps in output
            start_time: Start time in seconds (for segment selection)
            end_time: End time in seconds (for segment selection)
            progress_callback: Optional callback function for progress updates
                              Receives dict with keys: stage, progress, message, etc.

        Returns:
            Dictionary containing:
                - text: Full transcript text (with or without timestamps)
                - chunks: List of chunks with start/end times and text (if timestamps enabled)
                - duration: Audio duration in seconds
                - processing_time: Time taken to process
                - success: Boolean indicating success
                - error: Error message if failed
        """
        self.log(f"Transcribing: {audio_path}")
        start_processing_time = time.time()

        result = {
            'success': False,
            'text': '',
            'chunks': [],
            'duration': 0,
            'processing_time': 0,
            'error': None
        }

        try:
            # Stage 1: Loading audio
            if progress_callback:
                progress_callback({
                    'stage': 'loading',
                    'progress': 0.1,
                    'message': 'Loading audio file...'
                })

            audio, duration = self.load_audio_segment(audio_path, start_time, end_time)
            result['duration'] = duration

            # Stage 2: Transcribing
            if progress_callback:
                progress_callback({
                    'stage': 'transcribing',
                    'progress': 0.2,
                    'message': 'Starting transcription...'
                })

            self.log("Using pipeline with chunking for long-form transcription")

            # Use the pipeline which handles long-form audio properly
            pipe_result = self.pipe(audio)

            # Stage 3: Processing results
            if progress_callback:
                progress_callback({
                    'stage': 'processing',
                    'progress': 0.9,
                    'message': 'Processing transcription results...'
                })

            # Process results based on timestamp requirements
            if enable_timestamps and "chunks" in pipe_result:
                transcript_lines = []
                chunks_data = []

                for chunk in pipe_result["chunks"]:
                    timestamp = chunk.get("timestamp", (0, 0))
                    text = chunk.get("text", "").strip()

                    # Adjust timestamps if segment was selected
                    start = timestamp[0] if timestamp[0] is not None else 0
                    end = timestamp[1] if timestamp[1] is not None else duration

                    if start_time is not None:
                        start += start_time
                        end += start_time

                    start_ts = format_timestamp(start)
                    end_ts = format_timestamp(end)

                    transcript_lines.append(f"[{start_ts} - {end_ts}] {text}")
                    chunks_data.append({
                        'start': start,
                        'end': end,
                        'text': text
                    })

                result['text'] = "\n".join(transcript_lines)
                result['chunks'] = chunks_data
            else:
                result['text'] = pipe_result["text"]

            result['success'] = True
            result['processing_time'] = time.time() - start_processing_time

            # Stage 4: Complete
            if progress_callback:
                progress_callback({
                    'stage': 'complete',
                    'progress': 1.0,
                    'message': 'Transcription completed successfully'
                })

            self.log(f"Transcription completed in {result['processing_time']:.1f}s")
            return result

        except Exception as e:
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_processing_time
            self.log(f"Error transcribing {audio_path}: {str(e)}")

            if progress_callback:
                progress_callback({
                    'stage': 'error',
                    'progress': 0,
                    'message': f'Error: {str(e)}'
                })

            return result

    def save_transcript(
        self,
        result: Dict[str, Any],
        audio_path: str,
        output_dir: str,
        output_format: str = 'markdown'
    ) -> str:
        """
        Save transcription result to file.

        Args:
            result: Transcription result from transcribe_audio()
            audio_path: Original audio file path
            output_dir: Output directory
            output_format: 'markdown' or 'json'

        Returns:
            Path to saved file
        """
        audio_name = Path(audio_path).stem

        if output_format == 'json':
            output_file = Path(output_dir) / f"{audio_name}.json"
            self.log(f"Saving transcript to JSON: {output_file}")

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
            self.log(f"Saving transcript to Markdown: {output_file}")

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# Transcript: {audio_name}\n\n")
                f.write(f"**Source:** {Path(audio_path).name}\n\n")
                f.write("## Content\n\n")
                f.write(result['text'])
                f.write("\n")

        self.log(f"Transcript saved successfully")
        return str(output_file)
