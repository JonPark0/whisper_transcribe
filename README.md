# Whisper Transcriber

A Python tool that transcribes audio files using OpenAI's Whisper large-v3-turbo model. Supports multiple file formats, timestamp functionality, and AI-powered transcript enhancement.

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

### Advanced Usage
```bash
# With timestamps and verbose output
python3 convert.py -i "audio file.mp3" -o "./output folder/" -ts -v

# Multiple files with wildcards
python3 convert.py -i *.mp3 -o ./output/ -ts

# Multiple specific files
python3 convert.py -i file1.wav file2.mp3 file3.flac -o ./output/

# With timeout (300 seconds per file)
python3 convert.py -i *.mp3 -o ./output/ --timeout 300

# With chunked processing (60-second chunks) and Flash Attention
python3 convert.py -i audio.mp3 -o ./output/ -ch 60 --flash-attn

# Default chunked processing (30-second chunks)
python3 convert.py -i *.wav -o ./output/ -ch -ts -v

# With translation to English
python3 convert.py -i foreign_audio.mp3 -o ./output/ -tr en -ts -v

# Translate Spanish audio to English with Flash Attention
python3 convert.py -i spanish_lecture.mp3 -o ./output/ -tr en --flash-attn

# With automatic enhancement using Gemini API
python3 convert.py -i audio.mp3 -o ./output/ -ts -e

# With custom enhancement prompt
python3 convert.py -i lecture.mp3 -o ./output/ -e "Focus on technical terms and improve structure"
```

## Parameters

### Required Parameters
- `-i`, `--input`: Input audio file path(s). Supports wildcards and multiple files
- `-o`, `--output`: Output directory path for markdown files

### Optional Parameters
- `-ts`, `--timestamp`: Enable timestamp feature in transcription
- `-v`, `--verbose`: Enable verbose output to see processing details
- `-to`, `--timeout`: Set timeout in seconds for each file processing (Note: Not supported on Windows)
- `-ch`, `--chunked`: Enable chunked long-form processing with specified chunk length in seconds (default: 30)
- `--flash-attn`: Enable Flash Attention 2 for faster processing on compatible GPUs
- `-tr`, `--translate`: Set target language for translation using ISO 639-1 two-letter codes (e.g., "en", "es", "fr")
- `-e`, `--enhance`: Automatically execute enhancement process using Gemini API. Optional custom prompt can be provided
- `--json`: Output transcription in JSON format instead of Markdown
- `--start`: Start time in seconds for audio segment selection
- `--end`: End time in seconds for audio segment selection

## Output

The tool generates Markdown files (.md) in the specified output directory. Each output file has the same name as the input audio file but with a .md extension.

### Sample Output Format

```markdown
# Transcript: example_audio

**Source:** example_audio.mp3

## Content

[Without timestamps]
This is the transcribed content of the audio file...

[With timestamps (-ts flag)]
[00:00:00 - 00:00:05] This is the transcribed content
[00:00:05 - 00:00:10] of the audio file with timestamps...
```

## System Requirements

- Python 3.8 or higher
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
The enhancement feature uses Google's Gemini 2.0 Flash Experimental model to improve transcript quality:

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

**Usage:**
```bash
# Basic enhancement
python3 convert.py -i audio.mp3 -o ./output/ -e

# Enhancement with custom prompt
python3 convert.py -i lecture.mp3 -o ./output/ -e "Focus on technical accuracy"

# Enhancement with translation
python3 convert.py -i spanish_audio.mp3 -o ./output/ -tr en -e
```

**Standalone Enhancement:**
You can also enhance existing transcripts using `enhance.py` directly:
```bash
python3 enhance.py -i transcript.md -o enhanced.md -v
python3 enhance.py -i transcript.md -o enhanced.md -tr es
```

**Rate Limits:** Free tier supports 5 requests per minute. The tool handles rate limiting automatically.

## Notes

- First run will download the Whisper large-v3 model (~3GB)
- GPU acceleration is automatically used if available
- Processing time depends on audio length and hardware capabilities
- The tool handles various audio sample rates automatically
- MP3, WAV, and FLAC files work out of the box
- M4A/AAC files require system ffmpeg installation