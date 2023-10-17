# A demonstration of how Python can be made faster

## Cython

### OpenMPI

Newer versions of macOS does not support OpenMP in the clang compiler, so we need to install it ourselves.

```bash
brew install openmpi llvm libomp
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
```

```bash
poetry run python setup.py build_ext --inplace

poetry run python setup.py build_ext --inplace
```
