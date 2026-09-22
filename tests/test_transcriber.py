"""Tests for WhisperTranscriber class."""

import pytest
from core.transcriber import WhisperTranscriber


class TestWhisperTranscriber:
    """Tests for WhisperTranscriber class."""

    def test_initialization_default(self):
        """Test default initialization."""
        transcriber = WhisperTranscriber()
        assert transcriber.verbose is False
        assert transcriber.chunk_length == 30
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
