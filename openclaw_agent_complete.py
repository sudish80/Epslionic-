#!/usr/bin/env python3
"""
OpenClaw-Colab: Single-File LLM Training Agent (Thin Wrapper)
=============================================================
This file is a thin wrapper around the modular openclaw_colab_agent package.
It provides backward compatibility for Colab notebook usage.

Usage in Colab:
  !python openclaw_agent_complete.py --help
  !python openclaw_agent_complete.py --domain math
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    from openclaw_colab_agent.cli import main
    sys.exit(main())
