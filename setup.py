import setuptools

setuptools.setup(
    name="spins",
    version="0.2.0",
    python_requires=">=3.5",
    install_requires=[
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
        "scikit-umfpack",
        "scipy",
    ],
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
