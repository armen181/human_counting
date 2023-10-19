# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("tracking_module.pyx"),
    include_dirs=[np.get_include()]  # This line provides the necessary include directory
)
