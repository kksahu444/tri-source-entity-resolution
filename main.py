"""Convenience entry point: ``python main.py <command>`` works without installing the package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from entity_resolution.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
