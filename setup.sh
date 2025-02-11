#!/bin/bash
set -e  # Exit on error

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install build tools
pip install pyinstaller

# Install create-dmg using Homebrew if not already installed
if ! command -v create-dmg &> /dev/null; then
    brew install create-dmg
fi

# Install Python dependencies
pip install \
    PySide6 \
    openai==0.28.1 \
    tiktoken \
    gtts \
    pyqtdarktheme \
    pyautogui \
    pillow \
    anthropic \
    platformdirs \
    darkdetect

# Install the local computer-use-demo package in development mode
cd computer-use-demo
pip install -e .
cd ..

echo "Setup completed successfully!"

# for stupid async stuff
pip3 install --upgrade outcome trio

# Install 'pocketsphinx' to support voice-typing [optional]:
# pip3 install pocketsphinx
# note there are alternatives to PySide6 discussed in https://github.com/eliranwong/ChatGPT-GUI/wiki/Setup-%E2%80%90-macOS,-Linux,-ChromeOS

# chats is gitignored but we need it to exist for the app to work
# TODO in theory we should do this in python not here - done, this is in python now (and it puts it in the appropraite os-level folder for bundled release app)
# mkdir -p chats

# run
python3 mac-computer-use.py


