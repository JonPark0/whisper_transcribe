#!/usr/bin/env python3
"""Setup script for whisper_transcribe package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Core dependencies (required)
core_requirements = [
    "torch>=2.2.0",
    "transformers>=4.52.0",
    "librosa>=0.10.0",
    "ffmpeg-python>=0.2.0",
    "soundfile>=0.12.0",
    "pydub>=0.25.0",
]

# Optional dependencies for Flash Attention
flash_attn_requirements = [
    "ninja>=1.11.0",
    "psutil>=7.0.0",
    "flash-attn>=2.7.4",
]

# Optional dependencies for enhancement
enhancement_requirements = [
    "google-generativeai>=0.8.0",
    "python-dotenv>=1.0.0",
]

setup(
    name="whisper-transcribe",
    version="1.0.0",
    description="Audio transcription using OpenAI's Whisper large-v3-turbo model with optional Gemini enhancement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/JonPark0/whisper_transcribe",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "flash-attn": flash_attn_requirements,
        "enhancement": enhancement_requirements,
        "all": flash_attn_requirements + enhancement_requirements,
    },
    entry_points={
        "console_scripts": [
            "whisper-transcribe=convert:main",
            "whisper-enhance=enhance:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="whisper transcription audio speech-to-text ai gemini",
    include_package_data=True,
    zip_safe=False,
)
