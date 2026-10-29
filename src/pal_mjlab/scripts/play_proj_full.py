"""Play wrapper: adds --pretrained-checkpoint and --projector-backbone-normalize,
then defers to mjlab's play script."""

import os
import sys

import mjlab.scripts.play as play_mod


def _pop_flag(name: str) -> str | None:
  for i, arg in enumerate(sys.argv):
    if arg == name:
      if i + 1 >= len(sys.argv):
        raise ValueError(f"{name} requires a value")
      value = sys.argv[i + 1]
      del sys.argv[i : i + 2]
      return value
    if arg.startswith(name + "="):
      del sys.argv[i]
      return arg.split("=", 1)[1]
  return None


def _parse_bool(name: str, value: str) -> bool:
  v = value.strip().lower()
  if v in ("1", "true", "yes", "y"):
    return True
  if v in ("0", "false", "no", "n"):
    return False
  raise ValueError(f"{name} expects true/false, got '{value}'")


def main():
  ckpt = _pop_flag("--pretrained-checkpoint")
  if ckpt is not None:
    path = os.path.expanduser(ckpt)
    if not os.path.exists(path):
      raise FileNotFoundError(f"pretrained checkpoint not found: {path}")
    os.environ["PRETRAINED_CKPT"] = os.path.abspath(path)

  # Must match the value used for training: it decides where observations are
  # normalized (False = actor input with the saved running stats, True = inside
  # the frozen backbone with the base policy's stats).
  norm = _pop_flag("--projector-backbone-normalize")
  if norm is not None:
    os.environ["PROJECTOR_BACKBONE_NORMALIZE"] = (
      "1" if _parse_bool("--projector-backbone-normalize", norm) else "0"
    )

  play_mod.main()


if __name__ == "__main__":
  main()