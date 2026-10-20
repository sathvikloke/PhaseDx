"""Public CLI. Reading help never loads data or creates output directories."""
import argparse
import importlib
import json
import sys
from pathlib import Path

COMMANDS = {
    "manifest": "build_manifest28",
    "predict-image": "predict_image28",
    "predict-geometry": "predict_geometry28",
    "smoke": "smoke",
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    commands = tuple(COMMANDS) + ("acquire", "extract", "select", "lock", "receipt", "reference", "score", "flow", "export")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=commands)
    if not args or args[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    name = args.pop(0)
    if name in COMMANDS:
        return importlib.import_module("." + COMMANDS[name], __package__).main(args)
    if name in ("acquire", "extract"):
        from . import acquire31
        dry = "--dry-run" in args
        if dry:
            args.remove("--dry-run")
        return acquire31.main(["plan" if dry else "fetch" if name == "acquire" else "extract"] + args)
    if name in ("select", "lock", "receipt", "reference", "score", "flow", "export"):
        from . import commands
        return getattr(commands, name.replace("-", "_"))(args)
    parser.error("unknown command: " + name)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        sys.stderr.write(f"REFUSED: {exc}\n")
        sys.exit(2)
