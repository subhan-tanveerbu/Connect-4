"""
=============================================================================
main.py  –  Entry point for Connect-4  Human vs AI
=============================================================================
Run this file to start the game:

    python main.py

The script simply imports and calls gui.run_game(), which initialises
Pygame and starts the main event loop.
=============================================================================
"""

import sys
import os

# Ensure Python can find the src/ package regardless of where the user
# launches the script from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from gui import run_game

if __name__ == "__main__":
    run_game()
