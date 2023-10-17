# cython: infer_types=True
# distutils: language=c++
# cython: boundscheck = False

import cython
import numpy as np
from cython.parallel import parallel, prange
import cython.cimports.numpy as cnp

CUTOFF_T = cython.typedef(cnp.uint32_t)

cnp.import_array()

@cython.cfunc
@cython.nogil
@cython.boundscheck(False)
@cython.exceptval(-1)
def mandelbrot(x: cython.double, y: cython.double, cutoff: CUTOFF_T) -> CUTOFF_T:
    """Compute the margins of the mandelbrot set"""
    z = 0 + 0j
    c = x + y * 1j
    iterations = cython.declare(CUTOFF_T, 0)
    while iterations < cutoff and abs(z) <= 2:
        z = z ** 2 + c
        iterations += 1
    # The first iteration could be considered the zeroth, as z will always be 0
    # in that iteration, so the loop will be executed at least once.
    return iterations - 1

@cython.locals(
    x_scale=cython.double,
    y_scale=cython.double,
    i=cython.int,
    j=cython.int,
)
@cython.cdivision(True)
@cython.boundscheck(False)
def compute_mandelbrot(
        width: cython.int,
        height: cython.int,
        x: [cython.double, cython.double],
        y: [cython.double, cython.double],
        cutoff: CUTOFF_T,
):
    divergence: CUTOFF_T[:, ::1] = np.zeros((width, height), dtype=np.uint32)

    x_min: cython.double
    x_max: cython.double
    y_min: cython.double
    y_max: cython.double

    x_min, x_max = x
    y_min, y_max = y

    x_scale = abs(x_min - x_max) / width
    y_scale = abs(y_min - y_max) / height

    with cython.nogil, parallel():
        for i in prange(width):
            for j in prange(height):
                divergence[i, j] = mandelbrot(x_min + i * x_scale, y_min + j * y_scale, cutoff)
    return divergence
