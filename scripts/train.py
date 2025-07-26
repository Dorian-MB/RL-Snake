#!/usr/bin/env python3
"""Training script for RL Snake models."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rl_snake.agents.trainer import main

if __name__ == "__main__":
    main()
