#pylint: disable=C
import os
from setuptools import setup, find_packages

this_dir = os.path.dirname(__file__)

setup(
    name='util_cnn',
    packages=find_packages(exclude=["build"])
)
