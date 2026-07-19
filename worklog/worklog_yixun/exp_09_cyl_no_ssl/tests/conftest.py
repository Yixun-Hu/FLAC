"""pytest bootstrap for the exp-09 Stage A audit tests.

Puts the exp-09 runner directory on ``sys.path`` (so ``import audit_convention`` works) and
silences the benign RoPE aliasing / config-type warnings the tiny fixture models emit. The
runner itself adds the FLAC worktree root to ``sys.path`` at import time (for ``import src.*``).
"""
import sys
import warnings
from pathlib import Path

_EXP09_DIR = Path(__file__).resolve().parents[1]
if str(_EXP09_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP09_DIR))

warnings.filterwarnings("ignore", message=".*azimuth harmonics.*")
warnings.filterwarnings("ignore", message=".*not supported for all configurations.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
