import sys                                                     # adjust import path
from pathlib import Path                                        # locate repo root

sys.path.insert(0, str(Path(__file__).resolve().parent))       # make top-level packages importable in tests
