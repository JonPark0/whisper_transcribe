#!/usr/bin/env python3

import argparse
import os
import sys
import time
from pathlib import Path
import google.generativeai as genai
from typing import Optional
from dotenv import load_dotenv

class TranscriptEnhancer:
    def __init__(self, verbose: bool = False, target_language: str = None):
        self.verbose = verbose
        self.target_language = target_language
        self.model = None

        # Gemini 2.5 Pro token limits
        self.MAX_INPUT_TOKENS = 1_048_576  # ~4.2M characters
        self.MAX_OUTPUT_TOKENS = 65_536    # ~262K characters

    def log(self, message: str):
        if self.verbose:
            print(f"[INFO] {message}")

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using the 4-character approximation"""
        return len(text) // 4

    def format_token_count(self, tokens: int) -> str:
        """Format token count with commas and percentage of limit"""
        return f"{tokens:,}"

    def validate_input_size(self, content: str, prompt: str) -> tuple[bool, str]:
        """Validate if input content + prompt fits within token limits"""
        content_tokens = self.estimate_tokens(content)
        prompt_tokens = self.estimate_tokens(prompt)
        total_input_tokens = content_tokens + prompt_tokens

        self.log(f"Token analysis:")
        self.log(f"  Original content: {self.format_token_count(content_tokens)} tokens")
        self.log(f"  Enhancement prompt: {self.format_token_count(prompt_tokens)} tokens")
        self.log(f"  Total input: {self.format_token_count(total_input_tokens)} tokens")
        self.log(f"  Input limit: {self.format_token_count(self.MAX_INPUT_TOKENS)} tokens")
        self.log(f"  Input usage: {(total_input_tokens / self.MAX_INPUT_TOKENS * 100):.1f}%")

        if total_input_tokens > self.MAX_INPUT_TOKENS:
            excess_tokens = total_input_tokens - self.MAX_INPUT_TOKENS
            error_msg = (
                f"Input exceeds token limit by {self.format_token_count(excess_tokens)} tokens.\n"
                f"Total input: {self.format_token_count(total_input_tokens)} tokens\n"
                f"Maximum allowed: {self.format_token_count(self.MAX_INPUT_TOKENS)} tokens\n"
                f"Consider reducing the transcript size or splitting into smaller files."
            )
            return False, error_msg

        return True, ""

    def estimate_output_requirements(self, input_tokens: int) -> int:
        """Estimate required output tokens based on input size"""
        # Enhanced content is typically 1.2-1.5x the original size
        # due to improved grammar, structure, and formatting
        return int(input_tokens * 1.3)

    def validate_output_capacity(self, input_tokens: int) -> tuple[bool, str]:
        """Check if estimated output will fit within output token limits"""
        estimated_output = self.estimate_output_requirements(input_tokens)

        self.log(f"Output analysis:")
        self.log(f"  Estimated output: {self.format_token_count(estimated_output)} tokens")
        self.log(f"  Output limit: {self.format_token_count(self.MAX_OUTPUT_TOKENS)} tokens")
        self.log(f"  Output usage: {(estimated_output / self.MAX_OUTPUT_TOKENS * 100):.1f}%")

        if estimated_output > self.MAX_OUTPUT_TOKENS:
            excess_tokens = estimated_output - self.MAX_OUTPUT_TOKENS
            error_msg = (
                f"Estimated output may exceed token limit by {self.format_token_count(excess_tokens)} tokens.\n"
                f"Estimated output: {self.format_token_count(estimated_output)} tokens\n"
                f"Maximum allowed: {self.format_token_count(self.MAX_OUTPUT_TOKENS)} tokens\n"
                f"Consider splitting the transcript into smaller sections."
            )
            return False, error_msg

        return True, ""

    def setup_gemini(self, api_key: str):
        """Setup Gemini API with the provided API key"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-pro')
            self.log("Gemini 2.5 Pro API configured successfully")
        except Exception as e:
            raise Exception(f"Failed to configure Gemini API: {str(e)}")

    def get_default_prompt(self) -> str:
        """Get the default enhancement prompt"""
        base_prompt = """You are an expert transcript editor and enhancer. Please improve this audio transcript by:

1. **Grammar & Structure**: Fix grammatical errors, improve sentence structure, and ensure proper punctuation
2. **Clarity & Readability**: Make the text more readable while preserving the original meaning and speaker's intent
3. **Formatting**: Organize content with appropriate headings, bullet points, and paragraphs where suitable
4. **Coherence**: Ensure logical flow and smooth transitions between ideas
5. **Technical Terms**: Correct any misheard technical terms or jargon based on context
6. **Filler Words**: Remove excessive "um," "uh," "you know," etc. while keeping natural speech patterns
7. **Speaker Intent**: Maintain the original tone and meaning - don't add new information

Important guidelines:
- Preserve all factual content and key information
- Keep the original speaker's voice and style
- Don't add information that wasn't in the original transcript
- Maintain timestamps if present
- Focus on enhancement, not rewriting

Please provide the enhanced transcript:"""

        if self.target_language and self.target_language != "en":
            base_prompt += f"\n\nAdditionally, translate the enhanced transcript to {self.target_language} (ISO 639-1: {self.target_language})."

        return base_prompt

    def enhance_transcript(self, input_file: str, output_file: str, custom_prompt: str = None) -> str:
        """Enhance the transcript using Gemini API with pre-flight validation"""
        self.log(f"Reading transcript from: {input_file}")

        # Read the input markdown file
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            raise Exception(f"Failed to read input file: {str(e)}")

        # Prepare the prompt
        if custom_prompt:
            prompt = f"{custom_prompt}\n\nOriginal transcript:\n{original_content}"
        else:
            prompt = f"{self.get_default_prompt()}\n\nOriginal transcript:\n{original_content}"

        # Validate input size
        self.log("Validating token limits...")
        input_valid, input_error = self.validate_input_size(original_content, prompt)
        if not input_valid:
            raise Exception(f"Input validation failed:\n{input_error}")

        # Validate output capacity
        content_tokens = self.estimate_tokens(original_content)
        output_valid, output_error = self.validate_output_capacity(content_tokens)
        if not output_valid:
            print(f"⚠️  Warning: {output_error}")
            print("Proceeding anyway, but output may be truncated.")

        self.log("✅ Token validation passed. Sending to Gemini for enhancement...")
        start_time = time.time()

        try:
            # Generate enhanced content
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent, factual enhancement
                    max_output_tokens=self.MAX_OUTPUT_TOKENS,  # Use configured limit
                )
            )

            processing_time = time.time() - start_time
            enhanced_content = response.text

            # Validate actual output size
            actual_output_tokens = self.estimate_tokens(enhanced_content)
            self.log(f"Actual output: {self.format_token_count(actual_output_tokens)} tokens")

            if actual_output_tokens >= self.MAX_OUTPUT_TOKENS * 0.95:  # 95% threshold
                print("⚠️  Warning: Output is close to token limit. Content may be truncated.")

            self.log(f"Enhancement completed in {processing_time:.1f}s")

            # Save the enhanced content
            self._save_enhanced_content(enhanced_content, input_file, output_file, processing_time, actual_output_tokens)
            return enhanced_content

        except Exception as e:
            raise Exception(f"Failed to enhance transcript: {str(e)}")

    def _save_enhanced_content(self, enhanced_content: str, input_file: str, output_file: str, processing_time: float, output_tokens: int):
        """Save enhanced content with comprehensive metadata"""
        self.log(f"Saving enhanced transcript to: {output_file}")

        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add enhancement metadata with token information
        metadata = f"""# Enhanced Transcript

**Original File:** {Path(input_file).name}
**Enhanced:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Processing Time:** {processing_time:.1f}s
**Model:** Gemini 2.5 Pro
**Output Tokens:** {self.format_token_count(output_tokens)}
{'**Target Language:** ' + self.target_language if self.target_language else ''}

---

"""

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(metadata + enhanced_content)

        self.log("Enhanced transcript saved successfully")

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

def main():
    parser = argparse.ArgumentParser(
        description="Enhance audio transcripts using Google Gemini 2.5 Pro API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python3 enhance.py -i transcript.md -o enhanced.md
  python3 enhance.py -i transcript.md -o enhanced.md -v -tr es

  # Multiple files (batch processing)
  python3 enhance.py -i *.md -o enhanced/ -v
  python3 enhance.py -i file1.md file2.md file3.md -o output_dir/ -tr en

Requirements:
  - Google AI API key (set as GEMINI_API_KEY environment variable)
  - Install: pip install google-generativeai
        """
    )

    parser.add_argument('-i', '--input', nargs='+', required=True,
                       help='Input markdown file path(s). Supports multiple files.')
    parser.add_argument('-o', '--output', required=True,
                       help='Output directory path for enhanced files (when multiple inputs) or output file path (when single input).')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output to see processing details.')
    parser.add_argument('-tr', '--translate', type=str, metavar='LANGUAGE',
                       help='Set target language for translation using ISO 639-1 two-letter codes (e.g., "en", "es", "fr").')
    parser.add_argument('-p', '--prompt', type=str,
                       help='Custom enhancement prompt (replaces default prompt).')

    args = parser.parse_args()

    # Validate input files
    input_files = []
    for input_path in args.input:
        if os.path.isfile(input_path):
            input_files.append(input_path)
        else:
            print(f"Error: Input file '{input_path}' not found.")
            sys.exit(1)

    # Determine if processing single file or multiple files
    is_batch = len(input_files) > 1

    # Validate output path
    if is_batch:
        # Multiple inputs: output should be a directory
        output_dir = args.output
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        elif not os.path.isdir(output_dir):
            print(f"Error: For multiple inputs, output must be a directory: {output_dir}")
            sys.exit(1)
    else:
        # Single input: output can be a file
        output_file = args.output
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Load environment variables from .env file in project directory
    load_dotenv()

    # Suppress Google Cloud warnings when running locally
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY not found.")
        print("Please set your Google AI API key either:")
        print("1. Create a .env file in the project directory with: GEMINI_API_KEY=your-api-key-here")
        print("2. Or export as environment variable: export GEMINI_API_KEY='your-api-key-here'")
        sys.exit(1)

    # Create enhancer
    enhancer = TranscriptEnhancer(verbose=args.verbose, target_language=args.translate)

    try:
        total_start_time = time.time()

        # Setup Gemini API
        enhancer.setup_gemini(api_key)

        if is_batch:
            print(f"Processing {len(input_files)} files in batch mode...")
            enhanced_contents = []

            for i, input_file in enumerate(input_files, 1):
                print(f"\n[{i}/{len(input_files)}] Processing: {Path(input_file).name}")

                # Generate output filename
                input_name = Path(input_file).stem
                output_file = Path(output_dir) / f"{input_name}_enhanced.md"

                try:
                    # Enhance the transcript
                    enhanced_content = enhancer.enhance_transcript(
                        input_file=input_file,
                        output_file=str(output_file),
                        custom_prompt=args.prompt
                    )
                    enhanced_contents.append(enhanced_content)
                    print(f"✅ Saved: {output_file}")

                    # Rate limiting between files
                    if i < len(input_files):
                        print("⏳ Waiting 30 seconds before next file (rate limiting)...")
                        time.sleep(30)

                except Exception as e:
                    print(f"❌ Error processing {input_file}: {str(e)}")
                    continue

            total_time = time.time() - total_start_time
            successful_files = len(enhanced_contents)

            print(f"\n{'='*50}")
            print(f"✅ Batch Enhancement Complete")
            print(f"{'='*50}")
            print(f"📁 Input files: {len(input_files)}")
            print(f"✅ Successfully processed: {successful_files}")
            print(f"❌ Failed: {len(input_files) - successful_files}")
            print(f"📁 Output directory: {output_dir}")
            print(f"⏱️  Total time: {format_duration(total_time)}")
            if args.translate:
                print(f"🌍 Target language: {args.translate}")
            print(f"{'='*50}")

        else:
            # Single file processing
            input_file = input_files[0]
            output_file = args.output

            # Enhance the transcript
            enhanced_content = enhancer.enhance_transcript(
                input_file=input_file,
                output_file=output_file,
                custom_prompt=args.prompt
            )

            total_time = time.time() - total_start_time

            print(f"\n{'='*50}")
            print(f"✅ Enhancement Complete")
            print(f"{'='*50}")
            print(f"📄 Input: {input_file}")
            print(f"📄 Output: {output_file}")
            print(f"⏱️  Total time: {format_duration(total_time)}")
            if args.translate:
                print(f"🌍 Target language: {args.translate}")
            print(f"📊 Character count: {len(enhanced_content):,}")
            print(f"{'='*50}")

    except KeyboardInterrupt:
        print("\nEnhancement interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()