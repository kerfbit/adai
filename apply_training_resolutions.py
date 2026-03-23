#!/usr/bin/env python3
"""
Convenience launcher — apply review decisions to source training files.

Run from the project root:
    python apply_training_resolutions.py
    python apply_training_resolutions.py --dry-run
    python apply_training_resolutions.py --resolutions training_sessions/abnormal_resolutions.json \\
                                         --data-dir gutenberg_data/
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from tools.abnormal_review.apply_resolutions import main

if __name__ == "__main__":
    main()
