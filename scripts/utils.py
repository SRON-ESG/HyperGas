#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024 HyperGas developers
#
# This file is part of hypergas.
#
# hypergas is a library to retrieve trace gases from hyperspectral satellite data
"""Utils used in hypergas scripts."""

import os


def get_dirs(root_dir):
    """Get all lowest directories"""
    lowest_dirs = list()

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not d[0] == '.']
        if files and not dirs:
            lowest_dirs.append(root)

    return list(sorted(lowest_dirs))
