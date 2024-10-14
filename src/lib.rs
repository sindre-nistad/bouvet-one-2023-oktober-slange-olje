use ::pyo3::prelude::*;
use num::complex::{Complex, ComplexFloat};

/// Compute the margins of the mandelbrot set
#[pyfunction]
fn mandelbrot(x: f64, y: f64, cutoff: u32) -> PyResult<u32> {
    let mut z: Complex<f64> = Complex::new(0.0, 0.0);
    let c = Complex::new(x, y);
    let mut iterations = 0;
    while iterations < cutoff && z.abs() <= 2.0 {
        z = z.powu(2) + c;
        iterations += 1;
    }
    // The first iteration could be considered the zeroth, as z will always be 0
    // in that iteration, so the loop will be executed at least once.
    Ok(iterations - 1)
}

#[pymodule]
fn pyo3<'py>(
    module: &Bound<'py, PyModule>
) -> PyResult<()> {
    module.add_function(
        wrap_pyfunction!(mandelbrot, module)?
    )?;
    Ok(())
}
