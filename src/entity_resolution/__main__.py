"""Allow ``python -m entity_resolution <command>``."""
import sys

from entity_resolution.cli import main

sys.exit(main())
