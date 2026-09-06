# -*- coding: utf-8 -*-
"""
Project metadata that describe it
"""
package = 'fastgrab'
project = 'fastgrab'
project_no_spaces = project.replace(' ', '')
# Single source of truth is pyproject.toml; read it back from the installed
# distribution so this file can't drift. The literal is only a fallback for
# running from an un-installed source checkout.
try:
    from importlib.metadata import version as _dist_version
    version = _dist_version(package)
except Exception:  # PackageNotFoundError when not installed
    version = '0.4.0'
description = 'Low level screen capture package with a numpy interface'
authors = ['Mher Kazandjian']
authors_string = ', '.join(authors)
emails = ['mherkazandjian@gmail.com']
license = 'GPL v3'
copyright = '2019 ' + authors_string
url = 'https://github.com/mherkazandjian/fastgrab'
