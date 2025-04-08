# Contributing to DeepSearcher

We welcome contributions from everyone. This document provides guidelines to make the contribution process straightforward.


## Coding Style

Keeping a consistent style for code, code comments, commit messages, and PR descriptions will greatly accelerate your PR review process.
We highly recommend you run code linter and formatter when you put together your pull requests:

To check the coding styles:

```shell
make lint
```

To fix the coding styles:

```shell
make format
```

## Pull Request Process

1. Fork the repository and create your branch from `master`.
2. Make your changes.
3. Run tests and linting to ensure your code meets the project's standards.
4. Update documentation if necessary.
5. Submit a pull request.

## Developer Certificate of Origin (DCO)

All contributions require a sign-off, acknowledging the [Developer Certificate of Origin](https://developercertificate.org/). 
Add a `Signed-off-by` line to your commit message:

```text
Signed-off-by: Your Name <your.email@example.com>
```

## Development Environment Setup with UV

DeepSearcher uses [uv](https://github.com/astral-sh/uv) as the recommended package manager. UV is a fast, reliable Python package manager and installer.

### Setup with UV

1. Install uv if you haven't already:
   Follow the [offical installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

2. Create a virtual environment and install development dependencies:
   ```shell
   uv venv
   uv pip install -e ".[dev]"
   ```

3. To install all optional dependencies:
   ```shell
   uv pip install -e ".[all]"
   ```

4. To install specific optional dependencies:
   ```shell
   # Take optional `ollama` dependency
   uv pip install -e ".[ollama]"
   ```
   For more optional dependencies, refer to the `[project.optional-dependencies]` part of `pyproject.toml` file.

The project's pyproject.toml is configured to work with uv, which will provide faster dependency resolution and package installation compared to traditional tools.