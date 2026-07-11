# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.append(os.path.abspath("../../"))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import hypergas  # noqa: E402

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# get version using setuptools-scm
release = hypergas.__version__
# The full version, including alpha/beta/rc tags.
# for example take major/minor
version = ".".join(release.split(".")[:2])

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HyperGas'
copyright = u"2023-{}, The HyperGas Team".format(datetime.utcnow().strftime("%Y"))
author = 'The HyperGas Team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# sphinxcontrib.apidoc was added to sphinx in 8.2.0 as sphinx.etx.apidoc
needs_sphinx = "8.2.0"

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode', 'sphinx.ext.mathjax', "sphinx.ext.intersphinx", 'sphinx_rtd_theme', 'sphinx.ext.apidoc']

autoclass_content = "both"  # append class __init__ docstring to the class docstring

napoleon_use_rtype = False

# API docs
apidoc_modules = [
    {
        "path": "../../hypergas",
        "destination": "api",
        "exclude_patterns": [],
        'separate_modules': True,
    },
]

templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = ".rst"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  Major themes that come with
# Sphinx are currently 'default' and 'sphinxdoc'.
html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]

html_css_files = ["css/hypergas-landing.css",
                  "theme_overrides.css",
                  "https://cdn.datatables.net/1.10.23/css/jquery.dataTables.min.css",
                  "api_custom.css"]

html_show_sourcelink = False

html_js_files = [
    "https://cdn.datatables.net/1.10.23/js/jquery.dataTables.min.js",
    "main.js",
]

html_theme_options = {
    "logo": {
        "image_light": "../fig/logo_light.png",
        "image_dark": "../fig/logo_dark.png",
    },
    # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/header-links.html#fontawesome-icons
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/SRON-ESG/HyperGas",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_align": "content",
    "header_links_before_dropdown": 5,
}

project = "hypergas"

# The master toctree document.
master_doc = "index"

# allow dropdowns
collapse_navigation = False

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "dask": ("https://docs.dask.org/en/latest", None),
    "geoviews": ("https://geoviews.org", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/dev", None),
    "pyresample": ("https://pyresample.readthedocs.io/en/stable", None),
    "pytest": ("https://docs.pytest.org/en/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://scipy.github.io/devdocs", None),
    "xarray": ("https://xarray.pydata.org/en/stable", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/latest", None),
}
