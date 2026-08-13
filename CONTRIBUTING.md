# Contributing to InferX

Thank you for your interest in contributing to InferX!

## Prerequisites

Before getting started, ensure you have:

* Python 3.x
* CMake
* Ninja
* A C++ compiler with C++20 support
* Git

---

## Development Setup

Clone the repository and install InferX in development mode:

```bash
pip install .[dev]
```

You can then import InferX directly from Python:

```python
import inferX
```

### Useful Build Flags

Here is a list of flags that can be used with installation command:

| Flag                                | Purpose                                          |
| ----------------------------------- | ------------------------------------------------ |
| `-Ccmake.build-type=Debug`          | Enable debug builds and AddressSanitizer support |
| `-v`                                | verbose build output from `scikit-build-core`    |
| `-Ccmake.define.INFERX_USE_CUDA=ON` | build with cuda                                  |

---

## Pre-commit Hooks

Install the repository's pre-commit hooks:

```bash
pre-commit install
```

To run all hooks manually:

```bash
pre-commit run --all-files
```

---

## Building the C++ Component Independently

If you are only working on the native C++ layer, you can configure the build manually:

```bash
cmake \
  -G Ninja \
  -S .. \
  -B . \
  -DCMAKE_PREFIX_PATH=<absolute_path_to_pybind11> \
  -DCMAKE_BUILD_TYPE=Debug
```

Build:

```bash
cmake --build .
```

---

## Running Tests

Run the full test suite:

```bash
pytest
```

Run a specific test:

```bash
pytest path/to/test.py
```
