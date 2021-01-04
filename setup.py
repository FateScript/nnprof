#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import codecs
import os
import setuptools


def read(*parts):
    cur_path = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(cur_path, *parts), "r") as fp:
        return fp.read()


setuptools.setup(
    name="nnprof",
    version="0.1.0",
    author="Feng Wang",
    author_email="wffatescript@gmail.com",
    description="Profile tool for neural network(time, memory, etc.)",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/FateScript/nnprof",
    packages=setuptools.find_packages(),
    license="Apache License 2.0",
    install_requires=[
        "torch",
        "numpy",
    ],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
