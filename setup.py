from setuptools import find_packages
from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='PROJECT_NAME',
    version='1.0.0',
    author='AUTHOR_NAME',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='',
    description='',
    install_requires=required,
)
