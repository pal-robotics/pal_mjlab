"""Keep the documentation in sync with the registered tasks."""

import re
from pathlib import Path

import pal_mjlab.tasks  # noqa: F401  (registers the PAL tasks)
from mjlab.tasks.registry import list_tasks

DOCS_DIR = Path(__file__).resolve().parents[1] / "src" / "pal_mjlab" / "docs"


def _docs_text(include_history: bool = True) -> str:
  pages = DOCS_DIR.rglob("*.rst")
  if not include_history:
    # Release notes legitimately mention tasks that were removed.
    pages = (p for p in pages if p.name != "release_notes.rst")
  return "\n".join(p.read_text() for p in pages)


def test_every_registered_pal_task_is_documented():
  docs = _docs_text()
  pal_tasks = [t for t in list_tasks() if "-Pal-" in t]
  assert pal_tasks, "no PAL tasks registered"
  missing = [t for t in pal_tasks if f"``{t}``" not in docs and f"| {t} " not in docs]
  assert not missing, f"tasks missing from docs: {missing}"


def test_docs_do_not_reference_unregistered_pal_tasks():
  registered = set(list_tasks())
  documented = set(
    re.findall(
      r"Mjlab-[A-Za-z]+-[A-Za-z]+-Pal-[A-Za-z0-9_-]+", _docs_text(include_history=False)
    )
  )
  stale = sorted(documented - registered)
  assert not stale, f"docs reference unregistered tasks: {stale}"
