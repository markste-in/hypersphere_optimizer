# cython: language_level=3
from distutils.core import setup
from Cython.Build import cythonize
import sys

compiler_directives={'language_level' : sys.version_info[0]}

setup(
    ext_modules = cythonize("test_functions.py", compiler_directives=compiler_directives,
                            build_dir="build")
)