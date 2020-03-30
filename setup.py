import setuptools
import sys

extra_install_requires = []
if sys.version_info < (3, 7):
    # Use `dataclasses` package as a backport for Python 3.6.
    extra_install_requires += ["dataclasses"]

# Install Goos packages if using Python 3.6+.
if sys.version_info >= (3, 6):
    extra_install_requires += ["typing-inspect"]

setuptools.setup(
    name="spins",
    version="0.2.0",
    python_requires=">=3.5",
    install_requires=[
        "contours[shapely]",
        "dill",
        "flatdict",
        "gdspy>=1.4",
        "h5py",
        "jsonschema",
        "matplotlib",
        "numpy",
        "pandas",
        "pyyaml",
        "requests",
        "schematics",
        "scipy",
    ] + extra_install_requires,
    extras_require={
        "test": [
            "pytest",
            "pytest-cov",
            "pytest-xdist",
        ],
        "dev": [
            "pylint",
            "pytype",
            "yapf",
        ],
    },
    packages=setuptools.find_packages(),
)
