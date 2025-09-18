"""FractalSSL – modular research toolkit for fractal self-supervised learning."""

from importlib import metadata

__all__ = ["__version__"]


def __getattr__(name: str):
    if name == "__version__":
        try:
            return metadata.version("fractal-ssl")
        except metadata.PackageNotFoundError:  # pragma: no cover - dev installs
            return "0.0.0"
    raise AttributeError(name)
