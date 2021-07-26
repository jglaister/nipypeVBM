#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
setup

Setup for nipypeVBM package

Author: Jeffrey Glaister
"""
from glob import glob
from setuptools import setup, find_packages

args = dict(
    name='nipypeVBM',
    version='0.1',
    description='Runs fslvbm as a Nipype pipeline',
    author='Jeffrey Glaister',
    author_email='jeff.glaister@gmail.com',
    url='https://github.com/jglaister/nipypevbm'
)

setup(install_requires=['nipype', 'numpy', 'nibabel'],
      scripts=glob('bin/*'), **args)

