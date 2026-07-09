project = "pal_mjlab"
copyright = "2026, PAL Robotics"
author = "PAL Robotics"

extensions = [
  "sphinx.ext.mathjax",
]

source_suffix = {
  ".rst": "restructuredtext",
}

exclude_patterns = [
  "_build",
  "Thumbs.db",
  ".DS_Store",
]

language = "en"

html_title = "pal_mjlab Documentation"
html_theme = "sphinx_book_theme"
html_theme_options = {
  "repository_url": "https://github.com/pal-robotics/pal_mjlab",
  "use_repository_button": True,
}
