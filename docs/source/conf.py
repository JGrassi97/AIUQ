import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "AIUQ"
author = "Jacopo Grassi"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # supporto Google/NumPy docstring
    "support_matrix",
]

exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_title = "AIUQ · Weather & climate experiments"
html_short_title = "AIUQ"
html_static_path = ["_static"]
html_css_files = ["aiuq.css"]
html_favicon = "_static/favicon.svg"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 2,
    "style_nav_header_background": "#101f37",
}
html_show_sourcelink = False
html_show_sphinx = False
