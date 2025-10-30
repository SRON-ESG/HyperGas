#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""HyperGas package initializer"""

try:
    from hypergas.version import version as __version__
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "No module named hypergas.version. This could mean "
        "you didn't install 'hypergas' properly.")

from hypergas.hyper import Hyper
