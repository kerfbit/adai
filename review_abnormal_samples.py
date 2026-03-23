#!/usr/bin/env python3
"""
Convenience launcher — run from the project root:

    python review_abnormal_samples.py
    python review_abnormal_samples.py --samples path/to/other.json --no-resume
"""
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from tools.abnormal_review.app import main

if __name__ == "__main__":
    main()
