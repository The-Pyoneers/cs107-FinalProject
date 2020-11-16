# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import alabaster
sys.path.insert(0, os.path.abspath('../../farad/'))


# -- Project information -----------------------------------------------------

project = 'Farad'
copyright = '2020, M Stewart, X Ke, Z Hu, C Tseng'
author = 'M Stewart, X Ke, Z Hu, C Tseng'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "alabaster",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.6", None),
    "pyramid": (
        "https://docs.pylonsproject.org/projects/pyramid/en/1.9-branch",
        None,
    ),
    "cassandra": ("https://datastax.github.io/python-driver/", None),
    "pymemcache": ("https://pymemcache.readthedocs.io/en/latest/", None),
    "kazoo": ("https://kazoo.readthedocs.io/en/latest/", None),
    "kombu": ("https://kombu.readthedocs.io/en/latest/", None),
    "redis": ("https://redis-py.readthedocs.io/en/latest/", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/13/", None),
    "requests": ("https://requests.readthedocs.io/en/stable/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['source/_templates']

source_suffix = ['.md', '.rst']

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build','.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
language = "en"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_theme_path = [alabaster.get_path()]
html_static_path = ["source/_static"]


htmlhelp_basename = 'faraddoc'

html_theme_options = {
    "description": "Farad Automatic Differentiation Library",
    "github_button": False,
    "github_repo": "farad",
    "github_user": "The-Pyoneers",
    "github_banner": True,
    "logo": "faradlogo.png",
    "logo_name": True,
    "show_powered_by": False,
    "show_related": False,
    "show_relbars": True,
    "page_width": "960px",
}


# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "faradlogo.png"

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%b %d, %Y"

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# -- Autodoc --
autodoc_member_order = "bysource"


