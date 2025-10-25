#!/usr/bin/env python
"""Entry point wrapper for the tsalpha CLI.

Running this script is equivalent to invoking ``python -m
time_series_alpha_signal.cli``.  It exists for backwards
compatibility with the upstream repository and is used in some
examples.
"""

from time_series_alpha_signal.cli import main

if __name__ == "__main__":
    main()
