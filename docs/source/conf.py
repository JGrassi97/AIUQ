import os
import sys
sys.path.insert(0, os.path.abspath(".."))

project = "AIUQ"
author = "Jacopo Grassi"
release = "0.0.2"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # supporto Google/NumPy docstring
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"