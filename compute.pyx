# cython: infer_types=True
# distutils: language=c++
# cython: boundscheck = False

import cython
import numpy as np
import cython.cimports.numpy as cnp

CUTOFF_T = cython.typedef(cnp.uint32_t)

@np.vectorize
def mandelbrot(x: cython.double, y: cython.double, cutoff: CUTOFF_T) -> CUTOFF_T:
    """Compute the margins of the mandelbrot set"""
    z = 0 + 0j
    c = x + y * 1j
    iterations =  0
    while iterations < cutoff and z.real * z.real + z.imag * z.imag <= 4:
        z = z * z + c
        iterations += 1
    # The first iteration could be considered the zeroth, as z will always be 0
    # in that iteration, so the loop will be executed at least once.
    return iterations - 1


def compute_mandelbrot(width: cython.int, height: cython.int, x: tuple[cython.double, cython.double], y: tuple[cython.double, cython.double], cutoff: cython.int) -> CUTOFF_T[:, :]:
    x_scale = abs(x[0] - x[1]) / width
    y_scale = abs(y[0] - y[1]) / height

    x_inputs, y_inputs = np.indices((width, height), dtype=np.float64)
    divergence = mandelbrot(
        x_inputs * x_scale + x[0],
        y_inputs * y_scale + y[0],
        cutoff,
    )
    return divergence
