#!/usr/bin/env python3

import argparse
import os
import sys
import glob
import time
from pathlib import Path
import signal
from typing import List
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import librosa
import numpy as np
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout occurred")

class WhisperTranscriber:
    def __init__(self, verbose: bool = False, chunk_length: int = 30, use_flash_attn: bool = False, target_language: str = None):
        self.verbose = verbose
        self.chunk_length = chunk_length
        self.use_flash_attn = use_flash_attn
        self.target_language = target_language
        self.model = None
        self.processor = None
        self.pipe = None
        self.partial_transcript = ""
        self.processing_interrupted = False
        
    def log(self, message: str):
        if self.verbose:
            print(f"[INFO] {message}")
    
    def load_model(self):
        self.log("Loading Whisper large-v3 turbo model...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        model_id = "openai/whisper-large-v3-turbo"
        
        # Prepare model loading arguments
        model_kwargs = {
            "dtype": dtype,
            "low_cpu_mem_usage": True,
            "use_safetensors": True
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
        if self.target_language:
            generation_config["language"] = self.target_language
            generation_config["task"] = "translate"
            self.log(f"Translation enabled: translating to {self.target_language}")

        # Re-create pipeline for long-form transcription
        from transformers import pipeline
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=self.chunk_length,
            batch_size=16,
            return_timestamps=True,
            device=device,
            generate_kwargs=generation_config
        )
        self.log(f"Model loaded successfully on {device} with chunk length {self.chunk_length}s")
    
    def transcribe_audio(self, audio_path: str, enable_timestamps: bool = False) -> tuple[str, float]:
        self.log(f"Transcribing: {audio_path}")
        start_time = time.time()
        
        try:
            # Try to load audio with multiple methods for better format support
            audio = None

            # Try pydub first for better M4A/AAC support and to avoid librosa deprecation warnings
            if HAS_PYDUB:
                try:
                    self.log("Trying pydub for audio loading...")
                    audio_segment = AudioSegment.from_file(audio_path)
                    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                    audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                    if audio_segment.sample_width == 2:
                        audio = audio / 32768.0
                    elif audio_segment.sample_width == 4:
                        audio = audio / 2147483648.0
                    self.log(f"Successfully loaded audio with pydub. Duration: {len(audio)/16000:.2f}s")
                except Exception as e:
                    self.log(f"Pydub failed: {e}")
                    audio = None

            # Fallback to librosa if pydub fails
            if audio is None:
                try:
                    # Use librosa with soundfile backend to avoid audioread deprecation
                    import soundfile as sf
                    audio, sample_rate = sf.read(audio_path)
                    if sample_rate != 16000:
                        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                    # Convert to mono if stereo
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)
                    self.log(f"Successfully loaded audio with soundfile. Duration: {len(audio)/16000:.2f}s")
                except Exception as e:
                    self.log(f"Soundfile failed: {e}")
                    # Final fallback: librosa (may show deprecation warnings)
                    try:
                        audio, sample_rate = librosa.load(audio_path, sr=16000)
                        self.log(f"Successfully loaded audio with librosa (with warnings). Duration: {len(audio)/16000:.2f}s")
                    except Exception as e2:
                        raise Exception(f"All audio loading methods failed. Pydub: {e if HAS_PYDUB else 'Not available'}, Soundfile: {e}, Librosa: {e2}")
            
            if audio is None or len(audio) == 0:
                raise Exception("Audio file appears to be empty or corrupted")
            
            # Use pipeline for proper long-form transcription
            self.log("Using pipeline with chunking for long-form transcription")
            
            # Use the pipeline which handles long-form audio properly
            result = self.pipe(audio)
            
            # Process results based on timestamp requirements
            if enable_timestamps and "chunks" in result:
                transcript_lines = []
                for chunk in result["chunks"]:
                    timestamp = chunk.get("timestamp", (0, 0))
                    text = chunk.get("text", "")
                    start_timestamp = self.format_timestamp(timestamp[0]) if timestamp[0] is not None else "00:00:00"
                    end_timestamp = self.format_timestamp(timestamp[1]) if timestamp[1] is not None else "00:00:00"
                    transcript_lines.append(f"[{start_timestamp} - {end_timestamp}] {text.strip()}")
                transcription = "\n".join(transcript_lines)
            else:
                transcription = result["text"]
            
            processing_time = time.time() - start_time
            return transcription, processing_time
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.log(f"Error transcribing {audio_path}: {str(e)}")
            return f"Error: Could not transcribe {audio_path} - {str(e)}", processing_time
    
    def format_timestamp(self, seconds: float) -> str:
        if seconds is None:
            return "00:00:00"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def save_transcript(self, transcript: str, audio_path: str, output_dir: str):
        audio_name = Path(audio_path).stem
        output_file = Path(output_dir) / f"{audio_name}.md"
        
        self.log(f"Saving transcript to: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Transcript: {audio_name}\n\n")
            f.write(f"**Source:** {Path(audio_path).name}\n\n")
            f.write("## Content\n\n")
            f.write(transcript)
            f.write("\n")
        
        self.log(f"Transcript saved successfully")

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

def expand_file_paths(input_paths: List[str]) -> List[str]:
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
    audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
    return Path(file_path).suffix.lower() in audio_extensions

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
            
            transcript, processing_time = transcriber.transcribe_audio(audio_file, args.timestamp)
            transcriber.save_transcript(transcript, audio_file, args.output)
            processed_files += 1
            processing_times.append(processing_time)
            
            print(f"✅ Completed in {format_duration(processing_time)}")
            
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