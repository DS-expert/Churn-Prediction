from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent

if BASE_DIR not in sys.path:
    sys.path.append(str(BASE_DIR))