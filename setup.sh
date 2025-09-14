#!/bin/bash

echo "Setting up Whisper Transcriber..."
echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing core requirements..."
pip install -r requirements.txt

echo "Installing Flash Attention 2 (optional, for GPU acceleration)..."
echo "This may take 5-10 minutes to compile..."
pip install flash-attn --no-build-isolation || echo "Flash Attention 2 installation failed - continuing without it"

echo ""
echo "Setup complete!"
echo ""
echo "To use the transcriber:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the transcriber: python3 convert.py -i input.mp3 -o output_folder/"
echo ""
echo "For help: python3 convert.py --help"