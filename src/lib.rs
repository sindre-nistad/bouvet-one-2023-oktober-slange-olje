use ::pyo3::prelude::*;
use ::pyo3::Python;
use num::complex::{Complex, ComplexFloat};
use numpy::IntoPyArray;

/// Compute the margins of the mandelbrot set
fn mandelbrot(x: f64, y: f64, cutoff: u32) -> u32 {
    let mut z: Complex<f64> = Complex::new(0.0, 0.0);
    let c = Complex::new(x, y);
    let mut iterations = 0;
    while iterations < cutoff && z.abs() <= 2.0 {
        z = (z * z) + c;
        iterations += 1;
    }
    // The first iteration could be considered the zeroth, as z will always be 0
    // in that iteration, so the loop will be executed at least once.
    return iterations - 1
}

#[pyfunction]
fn compute_mandelbrot(py: Python, width: usize, height: usize, x: (f64, f64), y: (f64, f64), cutoff: u32) -> PyResult<Bound<numpy::PyArray2<u32>>> {
    let mut pixels = ndarray::Array2::<u32>::zeros((width, height));
    // let mut pixels = numpy::PyArray2::<f64>::zeros_bound(py, (width, height));
    let x_scale = num::abs(x.0 - x.1) / (width as f64);
    let y_scale = num::abs(y.0 - y.1) / (height as f64);

    for i in 0..width {
        for j in 0..height {
            pixels[[i, j]] = mandelbrot(x.0 + (i as f64) * x_scale, y.0 + (j as f64) * y_scale, cutoff)
        }
    };
    Ok(pixels.into_pyarray_bound(py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn pyo3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_mandelbrot, m)?)?;
    Ok(())
}
