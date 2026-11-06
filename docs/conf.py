"""Sphinx configuration for Array API Negative Index."""

from pathlib import Path
from typing import Any

from sphinx.application import Sphinx
from sphinx.ext import apidoc

project = "Array API Negative Index"
copyright = "2025, 34j"  # noqa: A001
author = "34j"
release = "1.0.0"

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]
napoleon_google_docstring = False
source_suffix = [".rst", ".md"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"


def generate_documentation_sources(app: Sphinx, _: Any) -> None:
    """Create the temporary landing page and changelog source."""
    docs_path = Path(app.srcdir)
    (docs_path / "index.md").write_text(
        """```{include} ../README.md

```

```{toctree}
:caption: Project Info
:maxdepth: 2
:glob:

*
```
""",
        encoding="utf-8",
    )
    (docs_path / "changelog.md").write_text("```{include} ../CHANGELOG.md\n```\n", encoding="utf-8")


def run_apidoc(_: Any) -> None:
    """Generate the API reference source document."""
    docs_path = Path(__file__).parent
    module_path = docs_path.parent / "src" / "array_api_negative_index"
    apidoc.main([
        "--force",
        "--module-first",
        "--no-toc",
        "-o",
        docs_path.as_posix(),
        module_path.as_posix(),
    ])


def cleanup_documentation_sources(app: Sphinx, _: Any) -> None:
    """Remove source files generated only for the documentation build."""
    for name in ("index.md", "changelog.md"):
        path = Path(app.srcdir) / name
        if path.exists():
            path.unlink()


def setup(app: Sphinx) -> None:
    """Register documentation generation hooks."""
    app.connect("config-inited", generate_documentation_sources)
    app.connect("builder-inited", run_apidoc)
    app.connect("build-finished", cleanup_documentation_sources)
