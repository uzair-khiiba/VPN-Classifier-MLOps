"""Fix a Windows-only bug in the installed Ludwig 0.11.0. Changes only .venv-ludwig, never the project.

ludwig/models/ecd.py opens the saved weights read-only and then calls os.fsync on that handle.
POSIX allows this, but Windows raises "[Errno 9] Bad file descriptor", which fails every trial
just as it finishes. Opening the file with "r+b" keeps the same behaviour and works on both.
It has to be patched in the file, because Ray runs each trial in a separate process.

Usage (safe to re-run): python experiments/patch_ludwig_windows.py
"""

from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / ".venv-ludwig/Lib/site-packages/ludwig/models/ecd.py"
OLD = 'with open(weights_save_path, "rb") as f:\n            os.fsync(f.fileno())'
NEW = 'with open(weights_save_path, "r+b") as f:\n            os.fsync(f.fileno())'


def main() -> None:
    source = TARGET.read_text(encoding="utf-8")
    if NEW in source:
        print("already patched")
    elif source.count(OLD) == 1:
        TARGET.write_text(source.replace(OLD, NEW), encoding="utf-8")
        print(f"patched {TARGET}")
    else:
        raise SystemExit(f"expected code not found in {TARGET}; is this Ludwig 0.11.0?")


if __name__ == "__main__":
    main()
