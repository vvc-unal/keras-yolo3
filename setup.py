#!/usr/bin/env python

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

setup(name='keras-yolo3',
      version='0.0.1',
      description='Keras implementation of Yolo v3',
      maintainer='Juan Navarro',
      maintainer_email='jsnavarroa@unal.edu.co',
      url='https://github.com/vvc-unal/keras-yolo3',
      packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required,
     )
