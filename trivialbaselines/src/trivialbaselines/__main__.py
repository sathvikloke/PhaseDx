"""``python -m trivialbaselines`` -- same entry point as the ``trivial-baselines`` script."""

from .core import main

if __name__ == "__main__":
    raise SystemExit(main())
