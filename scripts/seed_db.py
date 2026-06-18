"""
One-time database seeding script.

Usage (from project root):
    python -m scripts.seed_db
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.seeder import run_full_seed

if __name__ == "__main__":
    run_full_seed(n_recent=15)
