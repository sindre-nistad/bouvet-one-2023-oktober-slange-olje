# cython: infer_types=True
# distutils: language=c++
# cython: boundscheck = False

import cython
import numpy as np
from cython.parallel import parallel, prange
import cython.cimports.numpy as cnp
from cython.cimports.libc.math import floor


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
    while iterations < cutoff and z.real * z.real + z.imag * z.imag <= 4:
        z = z * z + c
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
        divergence: CUTOFF_T[:, ::1],
        x: [cython.double, cython.double],
        y: [cython.double, cython.double],
        cutoff: CUTOFF_T,
):
    width, height = divergence.shape[0:2]

    x_min: cython.double
    x_max: cython.double
    y_min: cython.double
    y_max: cython.double

    i: cython.int
    j: cython.int

    x_min, x_max = x
    y_min, y_max = y

    x_scale = abs(x_min - x_max) / width
    y_scale = abs(y_min - y_max) / height

    with cython.nogil, parallel():
        for i in prange(width):
            for j in prange(height):
                divergence[i, j] = mandelbrot(
                    x_min + i * x_scale,
                    y_min + j * y_scale,
                    cutoff,
                )


@cython.cdivision(True)
@cython.locals(
    n=cython.int,
    m=cython.int,

    i=cython.int,
    j=cython.int,
    k=cython.int,
    channel=cnp.uint8_t,
)
def apply_colormap(
    divergence: CUTOFF_T[:, ::1],
    cutoff: CUTOFF_T,
    colormap: cnp.uint8_t[:, ::1],
    pixels: cnp.uint8_t[:, :, ::1],
    # NB: cnp.uint8_t the _t is VERY important, as it denotes a C-object, and without you will get a Python object!
):

    num_colors = cython.declare(cython.double, (colormap.shape[0]))
    _cutoff = cython.declare(cython.double, cutoff)
    n, m = divergence.shape[0:2]

    with cython.nogil, parallel():
        for i in prange(n):
            for j in prange(m):
                color_index: cython.double = floor(divergence[i, j] / _cutoff * num_colors)
                for k in prange(3):
                    pixels[i, j, k] = colormap[cython.cast(cython.uint, color_index), k]
