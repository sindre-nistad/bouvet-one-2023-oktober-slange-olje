from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


ext_modules = [
    Extension(
        "*",
        ["compute.pyx"],
        extra_compile_args=[
            '-fopenmp',
            '-ffast-math',
        ],
        extra_link_args=['-lmpi', '-lomp', '-fopenmp'],
        include_dirs=[
            np.get_include(),
        ],
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ],
    )
]

setup(
    name="mandelbrot",
    ext_modules=cythonize(
        ext_modules,
        annotate=True,
    ),
)
