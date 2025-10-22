#!/usr/bin/env python3

import argparse
import os
import sys
import glob
import time
import signal
import subprocess
from pathlib import Path
from typing import List

# Import from core modules
from core import WhisperTranscriber, format_duration


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Timeout occurred")


def expand_file_paths(input_paths: List[str]) -> List[str]:
    """Expand wildcards and validate file paths."""
    expanded_paths = []
    for path in input_paths:
        if '*' in path or '?' in path:
            expanded_paths.extend(glob.glob(path))
        else:
            expanded_paths.append(path)

    valid_paths = []
    for path in expanded_paths:
        if os.path.isfile(path):
            valid_paths.append(path)
        else:
            print(f"Warning: File not found: {path}")

    return valid_paths


def is_audio_file(file_path: str) -> bool:
    """Check if file is a supported audio format."""
    audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
    return Path(file_path).suffix.lower() in audio_extensions


def enhance_file(audio_file: str, output_dir: str, custom_prompt: str, verbose: bool, translate: str):
    """Run enhancement on the generated transcript."""
    # Get the generated markdown file path
    audio_name = Path(audio_file).stem
    markdown_file = Path(output_dir) / f"{audio_name}.md"
    enhanced_file = Path(output_dir) / f"{audio_name}_enhanced.md"

    # Check if the enhance.py script exists
    enhance_script = Path(__file__).parent / "enhance.py"
    if not enhance_script.exists():
        raise Exception("enhance.py script not found in the same directory")

    # Build the command
    cmd = [sys.executable, str(enhance_script), "-i", str(markdown_file), "-o", str(enhanced_file)]

    if verbose:
        cmd.append("-v")

    if translate:
        cmd.extend(["-tr", translate])

    if custom_prompt:
        cmd.extend(["-p", custom_prompt])

    # Execute the enhancement
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if verbose:
            print(result.stdout)
        print(f"📄 Enhanced transcript saved: {enhanced_file}")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Enhancement process failed: {e.stderr}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI Whisper large-v3 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 convert.py -i audio.mp3 -o ./output/
  python3 convert.py -i "audio file.mp3" -o "./output folder/" -ts -v
  python3 convert.py -i *.mp3 -o ./output/ -ts --timeout 300
  python3 convert.py -i file1.wav file2.mp3 -o ./output/
  python3 convert.py -i audio.mp3 -o ./output/ -ch 60 --flash-attn
  python3 convert.py -i *.wav -o ./output/ -ch -ts -v
  python3 convert.py -i audio.mp3 -o ./output/ -tr en -ts -v
  python3 convert.py -i spanish_audio.mp3 -o ./output/ -tr en --flash-attn
  python3 convert.py -i audio.mp3 -o ./output/ -ts -e
  python3 convert.py -i audio.mp3 -o ./output/ -e "Focus on technical terms"
  python3 convert.py -i audio.mp3 -o ./output/ -ts --json  # JSON output
  python3 convert.py -i audio.mp3 -o ./output/ -ts --start 10 --end 60  # Segment selection
        """
    )

    parser.add_argument('-i', '--input', nargs='+', required=True,
                       help='Input audio file path(s). Supports wildcards and multiple files.')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory path for markdown files.')
    parser.add_argument('-ts', '--timestamp', action='store_true',
                       help='Enable timestamp feature in transcription.')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output to see processing details.')
    parser.add_argument('-to', '--timeout', type=int,
                       help='Set timeout in seconds for each file processing.')
    parser.add_argument('-ch', '--chunked', nargs='?', const=30, type=int,
                       help='Enable chunked long-form processing with specified chunk length in seconds (default: 30).')
    parser.add_argument('--flash-attn', action='store_true',
                       help='Enable Flash Attention 2 for faster processing on compatible GPUs.')
    parser.add_argument('-tr', '--translate', type=str, metavar='LANGUAGE',
                       help='Set target language for translation using ISO 639-1 two-letter codes (e.g., "en", "es", "fr").')
    parser.add_argument('-e', '--enhance', nargs='?', const='', type=str, metavar='PROMPT',
                       help='Automatically execute enhancement process using Gemini API. Optional additional prompt can be provided.')
    parser.add_argument('--json', action='store_true',
                       help='Output transcription in JSON format instead of Markdown.')
    parser.add_argument('--start', type=float, metavar='SECONDS',
                       help='Start time in seconds for audio segment selection.')
    parser.add_argument('--end', type=float, metavar='SECONDS',
                       help='End time in seconds for audio segment selection.')

    args = parser.parse_args()

    input_files = expand_file_paths(args.input)
    if not input_files:
        print("Error: No valid input files found.")
        sys.exit(1)

    audio_files = [f for f in input_files if is_audio_file(f)]
    if not audio_files:
        print("Error: No valid audio files found.")
        sys.exit(1)

    if len(audio_files) != len(input_files):
        print(f"Warning: {len(input_files) - len(audio_files)} non-audio files will be skipped.")

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        if args.verbose:
            print(f"Created output directory: {args.output}")

    # Set chunk length based on chunked parameter
    chunk_length = args.chunked if args.chunked is not None else 30

    transcriber = WhisperTranscriber(
        verbose=args.verbose,
        chunk_length=chunk_length,
        use_flash_attn=args.flash_attn,
        target_language=args.translate
    )
    transcriber.load_model()

    total_files = len(audio_files)
    processed_files = 0
    total_start_time = time.time()
    processing_times = []

    print(f"Processing {total_files} audio file(s)...")

    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{total_files}] Processing: {Path(audio_file).name}")

        try:
            if args.timeout:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(args.timeout)

            # Transcribe audio with segment selection support
            result = transcriber.transcribe_audio(
                audio_file,
                enable_timestamps=args.timestamp,
                start_time=args.start,
                end_time=args.end
            )

            if result['success']:
                # Determine output format
                output_format = 'json' if args.json else 'markdown'

                # Save transcript
                transcriber.save_transcript(result, audio_file, args.output, output_format)
                processed_files += 1
                processing_times.append(result['processing_time'])

                print(f"✅ Completed in {format_duration(result['processing_time'])}")

                # Run enhancement if requested
                if args.enhance is not None:
                    try:
                        print("🔧 Running enhancement...")
                        enhance_file(audio_file, args.output, args.enhance, args.verbose, args.translate)
                    except Exception as e:
                        print(f"⚠️  Enhancement failed: {str(e)}")
            else:
                print(f"❌ Error: {result['error']}")

            if args.timeout:
                signal.alarm(0)

        except TimeoutException:
            print(f"⏰ Timeout: Processing of {audio_file} exceeded {args.timeout} seconds")
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")
            break
        except Exception as e:
            print(f"❌ Error processing {audio_file}: {str(e)}")

    total_elapsed_time = time.time() - total_start_time

    print(f"\n{'='*50}")
    print(f"📊 Processing Summary")
    print(f"{'='*50}")
    print(f"✅ Processed: {processed_files}/{total_files} files")
    if processed_files < total_files:
        print(f"❌ Failed: {total_files - processed_files} files")

    print(f"⏱️  Total time: {format_duration(total_elapsed_time)}")

    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        print(f"📈 Average per file: {format_duration(avg_time)}")
        print(f"⚡ Fastest: {format_duration(min_time)}")
        print(f"🐌 Slowest: {format_duration(max_time)}")

    print(f"{'='*50}")


if __name__ == "__main__":
    main()
