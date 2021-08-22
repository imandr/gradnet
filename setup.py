#!/usr/bin/env python

import os
import numpy as np
from setuptools import setup, find_packages, Extension

def get_version():
    g = {}
    exec(open(os.path.join("gradnet", "version.py"), "r").read(), g)
    return g["Version"]

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

packages = ["gradnet", "gradnet/layers"]

setup(
    name = 'gradnet',
    version = get_version(),
    author = 'Igor Mandricnenko',
    author_email = "igorvm@gmail.com",
    description = "Machine learning framework",
    license = "BSD 3-clause",
    packages = packages,
    install_requires = ['numpy', 'scipy'],
    long_description=read('README.md'),
    ext_modules = [Extension('cconv',['cconv/cconv.c'])],
    zip_safe = False,
    include_dirs = [np.get_include()]
)
