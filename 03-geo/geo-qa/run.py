#!/usr/bin/env python3
"""
Startup script for OlmoEarth City Embeddings Web App
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import main

if __name__ == "__main__":
    main()
