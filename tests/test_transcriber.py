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
        assert transcriber.use_flash_attn is False
        assert transcriber.target_language is None
        assert transcriber.model is None
        assert transcriber.processor is None
        assert transcriber.pipe is None

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        transcriber = WhisperTranscriber(
            verbose=True,
            chunk_length=60,
            use_flash_attn=True,
            target_language="en"
        )
        assert transcriber.verbose is True
        assert transcriber.chunk_length == 60
        assert transcriber.use_flash_attn is True
        assert transcriber.target_language == "en"

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
