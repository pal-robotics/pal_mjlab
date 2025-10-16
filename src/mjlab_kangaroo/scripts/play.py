"""Registers the custom spinkick task before running mjlab's training pipeline."""

import mjlab_kangaroo.tasks  # noqa: F401
from mjlab.scripts.play import main

if __name__ == "__main__":
  main()