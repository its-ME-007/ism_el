import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Context-Aware Predictive Storage Tiering"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "sphinx_rtd_theme"
autodoc_typehints = "description"
