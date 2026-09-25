"""Tests for WhisperTranscriber class."""

import pytest
from core.transcriber import WhisperTranscriber


class TestWhisperTranscriber:
    """Tests for WhisperTranscriber class."""

    def test_initialization_default(self):
        """Test default initialization."""
        transcriber = WhisperTranscriber()
        assert transcriber.verbose is False
        assert transcriber.chunk_length == 0  # sequential long-form by default
        assert transcriber.batch_size == 16
        assert transcriber.use_flash_attn is False
        assert transcriber.target_language is None
        assert transcriber.model_id == "openai/whisper-large-v3-turbo"
        assert transcriber.language is None
        assert transcriber.temperature == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        assert transcriber.compression_ratio_threshold == 1.35
        assert transcriber.logprob_threshold == -1.0
        assert transcriber.no_speech_threshold == 0.6
        assert transcriber.model is None
        assert transcriber.processor is None
        assert transcriber.pipe is None

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        transcriber = WhisperTranscriber(
            verbose=True,
            chunk_length=60,
            batch_size=4,
            use_flash_attn=True,
            target_language="en",
            language="korean",
            temperature=0.0,
            compression_ratio_threshold=2.0,
            logprob_threshold=-0.5,
            no_speech_threshold=None
        )
        assert transcriber.verbose is True
        assert transcriber.chunk_length == 60
        assert transcriber.batch_size == 4
        assert transcriber.use_flash_attn is True
        assert transcriber.target_language == "en"
        assert transcriber.language == "korean"
        assert transcriber.temperature == 0.0
        assert transcriber.compression_ratio_threshold == 2.0
        assert transcriber.logprob_threshold == -0.5
        assert transcriber.no_speech_threshold is None

    def test_no_speech_threshold_requires_logprob_threshold(self):
        """no_speech_threshold alone would hit an UnboundLocalError deep inside
        transformers' fallback logic (logprobs is only bound in the
        logprob_threshold branch), so the constructor must reject it early."""
        with pytest.raises(ValueError, match="no_speech_threshold requires logprob_threshold"):
            WhisperTranscriber(no_speech_threshold=0.6, logprob_threshold=None)

    def test_thresholds_can_all_be_disabled(self):
        """None across the board must reproduce the pre-existing generation_config
        (no thresholds injected), for exact backward compatibility."""
        transcriber = WhisperTranscriber(
            temperature=None,
            compression_ratio_threshold=None,
            logprob_threshold=None,
            no_speech_threshold=None
        )
        assert transcriber.temperature is None
        assert transcriber.compression_ratio_threshold is None
        assert transcriber.logprob_threshold is None
        assert transcriber.no_speech_threshold is None

    def test_log_verbose_enabled(self, capsys):
        """Test logging with verbose enabled."""
        transcriber = WhisperTranscriber(verbose=True)
        transcriber.log("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_log_verbose_disabled(self, capsys):
        """Test logging with verbose disabled."""
        transcriber = WhisperTranscriber(verbose=False)
        transcriber.log("Test message")
        captured = capsys.readouterr()
        assert "Test message" not in captured.out

    # Note: We skip model loading tests as they require large downloads
    # and significant resources. Those should be integration tests.


class TestFasterWhisperProgress:
    """FasterWhisperTranscriber reports progress while consuming segments."""

    def test_progress_follows_segment_end_times(self, monkeypatch):
        from types import SimpleNamespace
        import numpy as np
        from core.faster_transcriber import FasterWhisperTranscriber

        t = FasterWhisperTranscriber()
        monkeypatch.setattr(t, "load_audio_segment", lambda *a, **k: (np.zeros(16000 * 100, dtype=np.float32), 100.0))

        def fake_transcribe(audio, **kwargs):
            def gen():
                for start in range(0, 100, 10):
                    yield SimpleNamespace(start=float(start), end=float(start + 10), text=f" s{start}")
            return gen(), SimpleNamespace(language="ko", language_probability=0.99)

        t.pipe = SimpleNamespace(transcribe=fake_transcribe)
        seen = []
        result = t.transcribe_audio("x.wav", enable_timestamps=True, progress_callback=seen.append)

        assert result["success"] is True
        assert len(result["chunks"]) == 10
        transcribing = [u["progress"] for u in seen if u["stage"] == "transcribing" and "%" in u["message"]]
        assert len(transcribing) == 10
        assert transcribing == sorted(transcribing)
        assert transcribing[-1] == pytest.approx(0.9)

    def test_translate_to_non_english_is_not_passed_as_language(self, monkeypatch):
        from types import SimpleNamespace
        import numpy as np
        from core.faster_transcriber import FasterWhisperTranscriber

        t = FasterWhisperTranscriber(language="ko", target_language="ja")
        monkeypatch.setattr(t, "load_audio_segment", lambda *a, **k: (np.zeros(16000, dtype=np.float32), 1.0))
        calls = []

        def fake_transcribe(audio, **kwargs):
            calls.append(kwargs)
            return iter([]), SimpleNamespace(language="ko", language_probability=0.99)

        t.pipe = SimpleNamespace(transcribe=fake_transcribe)
        assert t.transcribe_audio("x.wav")["success"] is True
        assert calls[0]["task"] == "transcribe"
        assert calls[0]["language"] == "ko"
