"""Setup script for epsionic."""
from setuptools import find_packages, setup

EXISTING = [p for p in find_packages(where=".") if not p.startswith((".", "build", "dist", "docs", "epsionic.egg-info"))]
setup(
    package_dir={
        "epsionic": ".",
        "epsionic.core": "core",
        "epsionic.tools": "tools",
        "epsionic.utils": "utils",
    },
    packages=["epsionic"] + [f"epsionic.{p}" for p in EXISTING],
    package_data={"": ["py.typed"]},
)
