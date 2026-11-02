"""Compatibility entry for an existing prepared e3_run.sh environment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spark_infer_q.cli import main


if __name__ == '__main__':
    raise SystemExit(main())
