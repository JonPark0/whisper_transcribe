"""Core transcription module with API and progress tracking support."""

import time
from typing import Optional, Callable, Dict, Any, List, Tuple, Union
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np

from .utils import format_timestamp, load_audio_segment, resolve_whisper_task, save_transcript


class WhisperTranscriber:
    """
    WhisperTranscriber provides audio transcription using OpenAI's Whisper model.

    This class can be used as a library (with progress callbacks) or through CLI.
    """

    def __init__(
        self,
        verbose: bool = False,
        chunk_length: int = 0,
        batch_size: int = 16,
        use_flash_attn: bool = False,
        target_language: Optional[str] = None,
        model_id: str = "openai/whisper-large-v3-turbo",
        language: Optional[str] = None,
        temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: Optional[float] = 1.35,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6
    ):
        """
        Initialize WhisperTranscriber.

        Args:
            verbose: Enable detailed logging
            chunk_length: 0 (default) = Whisper's sequential long-form
                          decoding: 30s windows decoded in order, each one
                          resuming at the previous window's last timestamp.
                          >0 = the pipeline's chunked mode: overlapping
                          windows of this length decoded in parallel and
                          stitched back together. Measured on 14.6 min of
                          non-repeating Korean speech: chunked (30) inserted
                          6.3% duplicated text at window seams (whole
                          sentences repeated), sequential matched the script
                          exactly, at similar speed (10.5x vs 9-12x realtime)
                          and lower VRAM (2.5 vs 3.4 GB peak).
            batch_size: Number of audio chunks to process simultaneously (default: 16).
                        Lower values reduce VRAM usage at the cost of speed.
            use_flash_attn: Enable Flash Attention 2 for faster GPU processing
            target_language: Translation target. Whisper can only translate
                             into English ("en"); any other value is ignored
                             here with a warning (the enhancer translates).
            model_id: HuggingFace repo id for the Whisper model
            language: Force recognition language (e.g. "korean", "english").
                      None = Whisper auto-detects. Distinct from target_language,
                      which triggers translation instead of transcription.
            temperature: Temperature(s) for the decoding fallback ladder. When a
                         segment fails compression_ratio_threshold or
                         logprob_threshold, transformers retries it at the next
                         temperature in this sequence. Pass a single float to
                         disable fallback (one decode attempt only).
            compression_ratio_threshold: Segments whose gzip compression ratio
                         exceeds this are treated as repetitive/low-quality and
                         trigger a temperature-fallback retry. None disables the
                         check. Typical value: 1.35.
            logprob_threshold: Segments whose average log-probability falls
                         below this trigger a temperature-fallback retry. None
                         disables the check. Typical value: -1.0.
            no_speech_threshold: When the "no speech" token probability exceeds
                         this AND logprob_threshold is also failed, the segment
                         is treated as silence/noise and its text is discarded
                         instead of retried. Requires logprob_threshold to be
                         set (see __init__ guard below). Typical value: 0.6.
        """
        if no_speech_threshold is not None and logprob_threshold is None:
            raise ValueError(
                "no_speech_threshold requires logprob_threshold to also be set: "
                "transformers' Whisper generation only reads the no-speech "
                "probability inside the logprob_threshold fallback branch "
                "(models/whisper/generation_whisper.py:_need_fallback); leaving "
                "logprob_threshold=None while setting no_speech_threshold raises "
                "an UnboundLocalError on the first low-confidence segment."
            )

        self.verbose = verbose
        self.chunk_length = chunk_length
        self.batch_size = batch_size
        self.use_flash_attn = use_flash_attn
        self.target_language = target_language
        self.model_id = model_id
        self.language = language
        self.temperature = temperature
        self.compression_ratio_threshold = compression_ratio_threshold
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
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

        model_id = self.model_id

        # Prepare model loading arguments. Default to SDPA attention explicitly
        # so the implementation does not silently change with transformers
        # versions; flash_attention_2 overrides it when requested.
        model_kwargs = {
            "dtype": dtype,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
            "attn_implementation": "sdpa"
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
        task, language, warning = resolve_whisper_task(self.language, self.target_language, self.model_id)
        if warning:
            print(f"[WARNING] {warning}")
        if task == "translate":
            generation_config["task"] = "translate"
            self.log("Translation enabled: translating to English")
        if language:
            generation_config["language"] = language
            generation_config.setdefault("task", "transcribe")
            self.log(f"Recognition language forced to {language}")

        # Decoding-robustness knobs (temperature fallback ladder + hallucination
        # filtering). Only inject the ones that are set so an all-None config
        # behaves exactly like the previous defaults.
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature
        if self.compression_ratio_threshold is not None:
            generation_config["compression_ratio_threshold"] = self.compression_ratio_threshold
        if self.logprob_threshold is not None:
            generation_config["logprob_threshold"] = self.logprob_threshold
        if self.no_speech_threshold is not None:
            generation_config["no_speech_threshold"] = self.no_speech_threshold
        if self.compression_ratio_threshold is not None or self.logprob_threshold is not None:
            self.log(
                f"Hallucination filtering enabled: compression_ratio_threshold="
                f"{self.compression_ratio_threshold}, logprob_threshold="
                f"{self.logprob_threshold}, no_speech_threshold={self.no_speech_threshold}"
            )

        # Re-create pipeline for long-form transcription
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=self.chunk_length or None,
            batch_size=self.batch_size,
            return_timestamps=True,
            device=device,
            generate_kwargs=generation_config
        )
        mode = f"chunked ({self.chunk_length}s windows)" if self.chunk_length else "sequential long-form"
        self.log(f"Model loaded successfully on {device} ({mode} decoding)")

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
        return load_audio_segment(audio_path, start_time, end_time, log=self.log)

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
            if self.pipe is None:
                raise RuntimeError("Transcriber pipeline is not initialized. Call load_model() first.")

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
        return save_transcript(result, audio_path, output_dir, output_format, log=self.log)
