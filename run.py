"""End-to-end run: generate corpus -> build indexes -> benchmark -> demo.

Usage: python run.py
Re-runnable: same seed -> same outputs (except wall-clock timings).
"""

import generate
import bench
import demo


def main() -> None:
    generate.main()
    bench.main()
    demo.main()


if __name__ == "__main__":
    main()
