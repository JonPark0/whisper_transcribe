"""CTranslate2-backed transcription engine (faster-whisper).

Same public interface as WhisperTranscriber (load_model, load_audio_segment,
transcribe_audio, save_transcript) so callers can swap engines without
touching call sites. Uses the same model weights as WhisperTranscriber's
default (openai/whisper-large-v3-turbo) via a CTranslate2 conversion -
faster-whisper is a different *runtime*, not a different model.

Measured through the actual whisper_service call path on production audio
(78.3 min, RTX 4070 SUPER): ~73x realtime solo on the GPU (~53x under
contention) vs. ~6x realtime for the transformers pipeline, with built-in VAD
that skips genuine silence before decoding (confirmed via faster-whisper's
own "VAD filter removed X of audio" log: 15-19% of that file) and
hallucination-filtering thresholds that don't retain score tensors on GPU
(unlike transformers' logprob_threshold path).

CAVEAT: this class always goes through BatchedInferencePipeline for the
batching speedup, which narrows the temperature fallback ladder to its first
value only and hardcodes condition_on_previous_text=False - see the
temperature/condition_on_previous_text docstrings on __init__ below. VAD and
the discard-only thresholds (compression_ratio/logprob/no_speech) are not
affected and remain the primary hallucination defense on this path.

Requires CUDA 12 runtime libraries even on a CUDA 13 host image - see
backend/Dockerfile's LD_LIBRARY_PATH for why.
"""

import time
from typing import Optional, Callable, Dict, Any, Tuple, Union

import numpy as np

from .utils import format_timestamp, load_audio_segment, resolve_whisper_task, save_transcript


class FasterWhisperTranscriber:
    """
    Audio transcription using faster-whisper (CTranslate2 runtime).

    Drop-in alternative to WhisperTranscriber. Same call signatures for
    transcribe_audio/load_audio_segment/save_transcript so whisper_service.py
    can select an engine without branching on call sites.
    """

    def __init__(
        self,
        verbose: bool = False,
        chunk_length: int = 30,
        batch_size: int = 16,
        use_flash_attn: bool = False,
        target_language: Optional[str] = None,
        model_id: str = "deepdml/faster-whisper-large-v3-turbo-ct2",
        language: Optional[str] = None,
        temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: Optional[float] = 1.35,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        vad_filter: bool = True,
        condition_on_previous_text: bool = False
    ):
        """
        Initialize FasterWhisperTranscriber.

        Args:
            verbose: Enable detailed logging
            chunk_length: Unused by faster-whisper (it VAD-segments internally
                          instead of fixed-length chunking); kept for
                          interface parity with WhisperTranscriber.
            batch_size: Batch size for BatchedInferencePipeline
            use_flash_attn: Unused - CTranslate2 has its own fused kernels and
                             does not use HF's attention implementations;
                             kept for interface parity.
            target_language: Translation target. Whisper can only translate
                             into English ("en"); any other value is ignored
                             here with a warning (the enhancer translates).
            model_id: CTranslate2 model repo id or local path. Defaults to a
                      float16 CT2 conversion of the same weights as
                      WhisperTranscriber's openai/whisper-large-v3-turbo
                      default (not a different model).
            language: Force recognition language (e.g. "ko", "en"; faster-whisper
                      uses ISO 639-1 codes, not the English names transformers
                      accepts). None = auto-detect per file (whole-file, unlike
                      the transformers pipeline's per-30s-chunk detection).
            temperature: Temperature(s) for the decoding fallback ladder.
                      CAVEAT (verified against faster-whisper 1.2.1's source):
                      BatchedInferencePipeline.transcribe - what this class
                      always uses, for the batching speedup - narrows this to
                      `temperature[:1]` internally; every value after the
                      first is silently dropped. Only WhisperModel's
                      unbatched transcribe() honors the full ladder. We still
                      accept and pass the full tuple here (a) so this class's
                      signature matches WhisperTranscriber's, and (b) in case
                      a future faster-whisper version threads it through
                      BatchedInferencePipeline. In practice, on this class,
                      only the first element takes effect - measured
                      end-to-end (78 min production audio) this still yields
                      near-zero repetition (2 of 761 sentences, both natural
                      conversational repeats, not decode loops), so we do not
                      work around it by dropping batching.
            compression_ratio_threshold: Segment discarded above this
                      compression ratio (no retry-at-higher-temperature, per
                      the caveat above - just flagged/dropped once). Typical
                      value: 1.35.
            logprob_threshold: Segment discarded below this average
                      log-probability (same caveat). Typical value: -1.0.
            no_speech_threshold: Segment treated as silence when the no-speech
                      probability exceeds this. Typical value: 0.6.
            vad_filter: Run Silero VAD before decoding and skip non-speech
                      spans entirely, instead of decoding them and relying on
                      no_speech_threshold to discard the result after the
                      fact. Primary lever against noise-triggered
                      hallucination, and NOT subject to the caveat above -
                      confirmed effective: 15-19% of the 78-minute production
                      file was correctly skipped as genuine silence.
            condition_on_previous_text: Accepted for interface parity with
                      WhisperTranscriber, but verified dead:
                      BatchedInferencePipeline.transcribe hardcodes
                      condition_on_previous_text=False internally regardless
                      of what's passed to it, and never reads this argument.
                      Harmless here since False is also our default, but if
                      you pass True expecting it to take effect, it will not.
        """
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
        self.vad_filter = vad_filter
        self.condition_on_previous_text = condition_on_previous_text

        self.model = None
        self.pipe = None
        self.device = None
        self.model_dtype = None

    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[INFO] {message}")

    def load_model(self):
        """Load the CTranslate2 Whisper model and wrap it in a batched pipeline."""
        import torch
        from faster_whisper import WhisperModel, BatchedInferencePipeline

        self.log(f"Loading faster-whisper model: {self.model_id}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        self.model = WhisperModel(self.model_id, device=device, compute_type=compute_type)
        self.pipe = BatchedInferencePipeline(model=self.model)

        self.device = device
        self.model_dtype = compute_type
        self.log(f"Model loaded successfully on {device} ({compute_type})")

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

        Same signature and result shape as WhisperTranscriber.transcribe_audio.
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
            if progress_callback:
                progress_callback({
                    'stage': 'loading',
                    'progress': 0.1,
                    'message': 'Loading audio file...'
                })

            audio, duration = self.load_audio_segment(audio_path, start_time, end_time)
            result['duration'] = duration

            if progress_callback:
                progress_callback({
                    'stage': 'transcribing',
                    'progress': 0.2,
                    'message': 'Starting transcription...'
                })

            if self.pipe is None:
                raise RuntimeError("Transcriber pipeline is not initialized. Call load_model() first.")

            # Same task/language mapping as WhisperTranscriber (Whisper only
            # translates into English; language names the source).
            task, language, warning = resolve_whisper_task(self.language, self.target_language, self.model_id)
            if warning:
                print(f"[WARNING] {warning}")
            if task == "translate":
                self.log("Translation enabled: translating to English")

            segments, info = self.pipe.transcribe(
                audio,
                batch_size=self.batch_size,
                language=language,
                task=task,
                vad_filter=self.vad_filter,
                condition_on_previous_text=self.condition_on_previous_text,
                temperature=list(self.temperature) if isinstance(self.temperature, (list, tuple)) else self.temperature,
                compression_ratio_threshold=self.compression_ratio_threshold,
                log_prob_threshold=self.logprob_threshold,
                no_speech_threshold=self.no_speech_threshold,
            )
            # The generator decodes lazily, batch by batch - consume it here
            # (it only runs once) and report progress by how far into the
            # audio the latest segment ends. Throttled to whole-percent steps
            # because callers like whisper_webui commit to a DB per update.
            collected = []
            last_reported = -1
            for seg in segments:
                collected.append(seg)
                if progress_callback and duration > 0:
                    pct = int(100 * min(seg.end / duration, 1.0))
                    if pct > last_reported:
                        last_reported = pct
                        progress_callback({
                            'stage': 'transcribing',
                            'progress': 0.2 + 0.7 * pct / 100,
                            'message': f'Transcribed {pct}% of audio'
                        })
            segments = collected

            if progress_callback:
                progress_callback({
                    'stage': 'processing',
                    'progress': 0.9,
                    'message': 'Processing transcription results...'
                })

            if enable_timestamps:
                transcript_lines = []
                chunks_data = []

                for seg in segments:
                    text = seg.text.strip()
                    start = seg.start
                    end = seg.end

                    if start_time is not None:
                        start += start_time
                        end += start_time

                    start_ts = format_timestamp(start)
                    end_ts = format_timestamp(end)

                    transcript_lines.append(f"[{start_ts} - {end_ts}] {text}")
                    chunks_data.append({'start': start, 'end': end, 'text': text})

                result['text'] = "\n".join(transcript_lines)
                result['chunks'] = chunks_data
            else:
                result['text'] = " ".join(seg.text.strip() for seg in segments)

            result['success'] = True
            result['processing_time'] = time.time() - start_processing_time

            if progress_callback:
                progress_callback({
                    'stage': 'complete',
                    'progress': 1.0,
                    'message': 'Transcription completed successfully'
                })

            self.log(
                f"Transcription completed in {result['processing_time']:.1f}s "
                f"(detected language: {info.language}, probability: {info.language_probability:.2f})"
            )
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
        """Save transcription result to file. See WhisperTranscriber.save_transcript."""
        return save_transcript(result, audio_path, output_dir, output_format, log=self.log)
