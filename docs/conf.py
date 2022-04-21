project = "spins"
copyright = "2018, Vuckovic Lab"
author = "Vuckovic Lab"

# The short X.Y version
version = ""

# The full version, including alpha/beta/rc tags
release = "0.2.0"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "matplotlib.sphinxext.plot_directive",
    "myst_parser",
    "nbsphinx",
    "sphinx_autodoc_typehints",
    "sphinx_markdown_tables",
]

napoleon_google_docstring = True
templates_path = ["_templates"]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}
language = "en"
myst_html_meta = {
    "description lang=en": "metadata description",
    "description lang=fr": "description des métadonnées",
    "keywords": "Sphinx, MyST",
    "property=og:locale": "en_US",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"
nbsphinx_timeout = 60
html_theme = "furo"
html_static_path = ["_static"]
htmlhelp_basename = "spinsdoc"
latex_documents = [
    (master_doc, "spins.tex", "spins Documentation", "Vuckovic Lab", "manual"),
]

man_pages = [(master_doc, "spins", "spins Documentation", [author], 1)]
texinfo_documents = [
    (
        master_doc,
        "spins",
        "spins Documentation",
        author,
        "spins",
        "One line description of project.",
        "Miscellaneous",
    ),
]
default_role = "code"
