"""Play wrapper: adds --pretrained-checkpoint, then defers to mjlab's play script."""

import os
import sys

import mjlab.scripts.play as play_mod


def _pop_flag(name: str) -> str | None:
  for i, arg in enumerate(sys.argv):
    if arg == name:
      value = sys.argv[i + 1]
      del sys.argv[i : i + 2]
      return value
    if arg.startswith(name + "="):
      del sys.argv[i]
      return arg.split("=", 1)[1]
  return None


def main():
  ckpt = _pop_flag("--pretrained-checkpoint")
  if ckpt is not None:
    path = os.path.expanduser(ckpt)
    if not os.path.exists(path):
      raise FileNotFoundError(f"pretrained checkpoint not found: {path}")
    os.environ["PRETRAINED_CKPT"] = os.path.abspath(path)
  play_mod.main()


if __name__ == "__main__":
  main()