from setuptools import setup, find_packages

import os
lib_folder = os.path.dirname(os.path.realpath(__file__))
install_requires = []

setup(
    name="very_simple_compling_demo_tools",
    author='ZP Rosen',
    version="0",
    setup_requires=install_requires,
    install_requires=install_requires,
    long_description="full package info available at the github repo: https://github.com/zaqari/DemoTools",
    long_description_content_type='text/markdown'
)
