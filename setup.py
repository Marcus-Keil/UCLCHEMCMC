# setup.py
import setuptools

from setuptools.command.install import install

import os


setuptools.setup(
        name="UCLCHEMCMC",
        version="0.1",
        author="Marcus Keil",
        author_email="marcus.keil.19@ucl.ac.uk",
        description="TBD",
        package_dir={'': 'UCLCHEMCMC/src'},
        classifiers=["Programming Language :: Python :: 3",
                     "License :: MIT License",
                     "Operating System :: Ubuntu"],
        packages=setuptools.find_packages(),
        install_requires=['pandas', 'numpy', 'corner', 'matplotlib', 'emcee',
                          'billiard', 'bokeh', 'flask', 'celery']
        )


