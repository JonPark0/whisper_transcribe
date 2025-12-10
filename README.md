# Whisper Transcribe

A Python tool that transcribes audio files using OpenAI's Whisper large-v3-turbo model. Supports multiple file formats, timestamp functionality, and AI-powered transcript enhancement.

This README provides a basic overview. For detailed instructions, please see the full **[documentation](docs/en/guide.md)**.

## Related Projects

- **[whisper_webui](https://github.com/JonPark0/whisper_webui):** A web-based user interface for `whisper_transcribe`, allowing you to manage transcription jobs easily through a browser.

---

## Features

- Uses OpenAI's Whisper large-v3-turbo model for high-quality transcription
- Supports multiple audio formats (MP3, WAV, FLAC, AAC, OGG, M4A, WMA)
- Multilingual transcription support
- Optional timestamp generation
- Batch processing with wildcard support
- Configurable timeout for processing
- Outputs transcriptions in Markdown format

## Setup

1. Clone or download this repository
2. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

Or manually:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
source venv/bin/activate
python3 convert.py -i input_audio.mp3 -o output_folder/
```

For advanced usage and detailed parameter explanations, please refer to the **[full documentation](docs/en/guide.md)**.

### Advanced Usage Examples
```bash
# With timestamps and verbose output
python3 convert.py -i "audio file.mp3" -o "./output folder/" -ts -v

# Multiple files with wildcards
python3 convert.py -i *.mp3 -o ./output/ -ts

# With automatic enhancement using Gemini API
python3 convert.py -i audio.mp3 -o ./output/ -ts -e
```

## System Requirements

- Python 3.9 or higher
- CUDA-compatible GPU (optional, for faster processing)
- Sufficient disk space for model downloads (~3GB for Whisper large-v3)

### For M4A/AAC Support (Optional)
If you need to process M4A/AAC files, install system ffmpeg:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH

## Dependencies

### Core Dependencies
- transformers>=4.52.0
- torch>=2.2.0  
- librosa>=0.10.0
- pydub>=0.25.0
- ffmpeg-python>=0.2.0
- soundfile>=0.12.0

### Optional Performance Dependencies
- ninja>=1.11.0 (build system for Flash Attention)
- psutil>=7.0.0 (system utilities for Flash Attention)
- flash-attn>=2.7.4 (Flash Attention 2 for GPU acceleration)

### Optional Enhancement Dependencies
- google-generativeai>=0.8.0 (Google Gemini API for transcript enhancement)

### Flash Attention 2 Requirements (Optional)
Flash Attention 2 provides significant performance improvements for GPU processing:

**GPU Requirements:**
- NVIDIA GPUs: Ampere, Ada, or Hopper architecture (RTX 3090, RTX 4070/4090, A100, H100, etc.)
- CUDA >= 12.3 (recommended: CUDA 12.8)
- AMD GPUs: MI200 or MI300 with ROCm 6.0+

**Installation (Method 1: Using setup.py extras):**
```bash
# Recommended: Install with optional flash-attn extras
pip install -e .[flash-attn]
```

**Installation (Method 2: Manual installation):**
```bash
# Step 1: Install core dependencies first (includes torch)
pip install -r requirements.txt

# Step 2: Install build tools
pip install ninja psutil

# Step 3: Install Flash Attention 2 (requires torch to be already installed)
pip install flash-attn --no-build-isolation
```

**Important Notes:**
- Flash Attention 2 installation may take 5-10 minutes as it compiles from source
- **torch must be installed before installing flash-attn** (flash-attn needs torch during build)
- If installation fails, the tool will automatically fall back to standard attention
- Not required for basic functionality - only provides performance optimization

### Transcript Enhancement with Gemini API (Optional)
The enhancement feature uses Google's Gemini API to improve transcript quality:

**Setup:**
1. Get a Google AI API key from [Google AI Studio](https://aistudio.google.com/)
2. Set the environment variable:
   ```bash
   export GEMINI_API_KEY='your-api-key-here'
   ```
3. Install the dependency:
   ```bash
   pip install google-generativeai>=0.8.0
   ```

**Enhancement Features:**
- Grammar and punctuation correction
- Improved sentence structure and readability
- Technical term correction based on context
- Removal of excessive filler words
- Better formatting with headings and structure
- Optional translation to target language

**Available Models:**
You can choose from different Gemini models based on your needs:
- `gemini-2.5-flash` (default) - Free tier available, fast processing, limited requests
- `gemini-2.5-flash-lite` - Free tier available, faster and lighter, limited requests
- `gemini-2.5-pro` - Paid tier, highest quality, no rate limits

**Usage:**
```bash
# Basic enhancement (uses gemini-2.5-flash by default)
python3 convert.py -i audio.mp3 -o ./output/ -e

# Enhancement with custom prompt
python3 convert.py -i lecture.mp3 -o ./output/ -e "Focus on technical accuracy"

# Enhancement with translation
python3 convert.py -i spanish_audio.mp3 -o ./output/ -tr en -e
```

**Standalone Enhancement:**
You can also enhance existing transcripts using `enhance.py` directly with model selection:
```bash
# Basic enhancement with default model
python3 enhance.py -i transcript.md -o enhanced.md -v

# Enhancement with translation
python3 enhance.py -i transcript.md -o enhanced.md -tr es

# Using specific Gemini models
python3 enhance.py -i transcript.md -o enhanced.md -m gemini-2.5-flash
python3 enhance.py -i transcript.md -o enhanced.md -m gemini-2.5-flash-lite
python3 enhance.py -i transcript.md -o enhanced.md -m gemini-2.5-pro

# Batch processing with specific model
python3 enhance.py -i *.md -o enhanced/ -m gemini-2.5-pro -v
```

**Rate Limits:**
- Free tier (gemini-2.5-flash, gemini-2.5-flash-lite): 5 requests per minute
- Paid tier (gemini-2.5-pro): Higher rate limits based on your plan
- The tool handles rate limiting automatically for batch processing

## Notes

- First run will download the Whisper large-v3 model (~3GB)
- GPU acceleration is automatically used if available
- For M4A/AAC support, system ffmpeg installation is required.
