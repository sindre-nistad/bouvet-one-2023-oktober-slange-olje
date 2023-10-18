# Setup

`numba` (0.60) krever LLVM 14 for å kompilere.
Dette må gjøres dersom maksinen ikke er x86.

```bash
asdf install
brew install llvm@14
export PATH="$(brew --prefix llvm@14)/bin:$PATH"
python -m venv venv.numba
source venv.numba/bin/activate
poetry install
```
