#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from collections import namedtuple

LeafModule = namedtuple("LeafModule", ["path", "module"])


def leaf_modules_generator(module, name=None, path=None):
    """
    Generate all leaf modules of an given pytorch module

    Args:
        module (nn.Module): a pytorch nn.Module
        name (string): name of pytorch module
        path (Tuple[string]): path to pytorch module

    Return:
        Generator contains LeafModule.
    """
    if path is None:
        path = ()

    if not name:
        name = module.__class__.__name__
    named_children = list(module.named_children())
    path = path + (name,)
    if len(named_children) == 0:
        yield LeafModule(".".join(path), module)
    # recursively walk into all submodules
    for name, child_module in named_children:
        yield from leaf_modules_generator(child_module, name=name, path=path)


def format_memory(nbytes):
    """Returns a formatted memory size string"""
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    if (abs(nbytes) >= GB):
        return '{:.2f} Gb'.format(nbytes * 1.0 / GB)
    elif (abs(nbytes) >= MB):
        return '{:.2f} Mb'.format(nbytes * 1.0 / MB)
    elif (abs(nbytes) >= KB):
        return '{:.2f} Kb'.format(nbytes * 1.0 / KB)
    else:
        return str(nbytes) + ' b'
