# Whisper Transcribe

A Python tool that transcribes audio files using OpenAI's Whisper large-v3-turbo model. Supports multiple file formats, timestamp functionality, and AI-powered transcript enhancement.

This README provides a basic overview. For detailed instructions, please see the full **[documentation](docs/en/guide.md)**.

## Related Projects

- **[whisper_webui](../whisper_webui/):** A web-based user interface for `whisper_transcribe`, allowing you to manage transcription jobs easily through a browser.

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

## Notes

- First run will download the Whisper large-v3 model (~3GB)
- GPU acceleration is automatically used if available
- For M4A/AAC support, system ffmpeg installation is required.