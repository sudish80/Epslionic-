"""Setup script for epsionic."""
from setuptools import find_packages, setup

setup(
    package_dir={
        "": ".",
        "epsionic.core": "core",
        "epsionic.tools": "tools",
        "epsionic.utils": "utils",
    },
    packages=find_packages(where=".") + ["epsionic.core", "epsionic.tools", "epsionic.utils"],
    package_data={"": ["py.typed"]},
)
