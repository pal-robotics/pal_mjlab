#!/usr/bin/env python3
"""Copy only the asset files an MJCF model actually references.

The upstream asset trees (e.g. kangaroo_simulation) hold every mesh ever
exported plus symlinks, so they cannot be copied wholesale. This parses a
model, collects every ``file=`` attribute under ``<asset>``, and copies just
those from a source asset root into the model's own asset directory,
preserving the relative paths so the XML keeps resolving unchanged.

Example::

    python sync_model_assets.py \
        --xml src/pal_mjlab/robots/pal_kangaroo_full/xmls/kangaroo_full.xml \
        --src-assets src/kangaroo_simulation/kangaroo_mujoco/models/assets \
        --clean
"""

from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Asset tags that take a ``file`` attribute, mapped to the compiler option
# giving their base directory. ``assetdir`` overrides both mesh and texture.
_DIR_ATTR = {
  "mesh": "meshdir",
  "skin": "meshdir",
  "hfield": "meshdir",
  "texture": "texturedir",
  "model": "meshdir",
}


def collect_references(xml_path: Path) -> dict[str, list[str]]:
  """Return {subdir_key: [relative file paths]} referenced by the model."""
  root = ET.parse(xml_path).getroot()

  dirs = {"meshdir": "", "texturedir": ""}
  for compiler in root.iter("compiler"):
    assetdir = compiler.get("assetdir")
    if assetdir is not None:
      dirs["meshdir"] = dirs["texturedir"] = assetdir
    for key in dirs:
      value = compiler.get(key)
      if value is not None:
        dirs[key] = value

  refs: dict[str, list[str]] = {}
  for asset in root.iter("asset"):
    for elem in asset:
      file_attr = elem.get("file")
      if file_attr is None or elem.tag not in _DIR_ATTR:
        continue
      base = dirs[_DIR_ATTR[elem.tag]]
      refs.setdefault(base, []).append(file_attr)
  return refs


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--xml", type=Path, required=True, help="MJCF model to parse")
  parser.add_argument(
    "--src-assets",
    type=Path,
    required=True,
    help="source asset root the model's file= paths are relative to",
  )
  parser.add_argument(
    "--dest",
    type=Path,
    default=None,
    help="destination asset root (default: <xml dir>/<meshdir>)",
  )
  parser.add_argument(
    "--clean",
    action="store_true",
    help="wipe the destination root before copying",
  )
  parser.add_argument(
    "--with-siblings",
    nargs="*",
    default=[],
    metavar="NAME",
    help="also copy these files from each asset's source directory "
    "(e.g. material.mtl), when present",
  )
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()

  xml_path = args.xml.resolve()
  src_root = args.src_assets.resolve()
  if not xml_path.is_file():
    parser.error(f"model not found: {xml_path}")
  if not src_root.is_dir():
    parser.error(f"source asset root not found: {src_root}")

  refs = collect_references(xml_path)
  if not refs:
    print("no file-backed assets referenced; nothing to do")
    return 0

  # All base dirs resolve under the model directory; the model's own asset
  # root is the shared prefix, which for a single base dir is just that dir.
  bases = set(refs)
  if args.dest is not None:
    dest_root = args.dest.resolve()
  elif len(bases) == 1:
    dest_root = (xml_path.parent / bases.pop()).resolve()
  else:
    parser.error(f"model uses multiple asset dirs {sorted(bases)}; pass --dest")

  wanted: list[str] = sorted({p for paths in refs.values() for p in paths})
  print(f"model:  {xml_path}")
  print(f"source: {src_root}")
  print(f"dest:   {dest_root}")
  print(f"referenced assets: {len(wanted)}")

  missing = [rel for rel in wanted if not (src_root / rel).is_file()]
  if missing:
    print(
      f"\nERROR: {len(missing)} referenced file(s) absent from source:", file=sys.stderr
    )
    for rel in missing:
      print(f"  {rel}", file=sys.stderr)
    return 1

  if args.clean and dest_root.exists():
    print(f"cleaning {dest_root}")
    if not args.dry_run:
      shutil.rmtree(dest_root)

  copied = 0
  for rel in wanted:
    src = src_root / rel
    dst = dest_root / rel
    sources = [src] + [
      src.parent / name for name in args.with_siblings if (src.parent / name).is_file()
    ]
    for s in sources:
      d = dst.parent / s.name
      if args.dry_run:
        print(f"  would copy {s.relative_to(src_root)}")
      else:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)  # follows symlinks
      copied += 1

  total = (
    sum(f.stat().st_size for f in dest_root.rglob("*") if f.is_file())
    if not args.dry_run
    else 0
  )
  print(
    f"{'would copy' if args.dry_run else 'copied'} {copied} file(s)"
    + (f" ({total / 1e6:.1f} MB)" if not args.dry_run else "")
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
