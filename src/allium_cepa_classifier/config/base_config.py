from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel


def find_project_root() -> Path:
    """Walk up from the current working directory until pyproject.toml is found.

    Using cwd (not __file__) so that editable installs shared across clones
    always resolve to the project that is actually being run, not the one where
    the package source happens to live on disk.
    """
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Project root not found: no pyproject.toml in any parent directory.")


class BaseConfig(BaseModel):
    """Base class for all config objects. Provides shared YAML loading."""

    @classmethod
    def from_yaml(cls, path: Path | str, key: str | None = None) -> Self:
        """
        Load config from a YAML file.

        Parameters
        ----------
        path : Path | str
            Path to the YAML file.
        key : str | None
            If provided, read from data[key] instead of the root.
            Useful when multiple configs share a single YAML file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(**(data[key] if key else data))
