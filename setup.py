#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
    "numpy",
    "scipy",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Ryuichi Shimogawa",
    author_email="ryuichi@shimogawa.com",
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    description="Iterative Bragg Peak removal from X-ray absoption spectra",
    entry_points={
        "console_scripts": [
            "ibr_xas=ibr_xas.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="ibr_xas",
    name="ibr_xas",
    packages=find_packages(include=["ibr_xas", "ibr_xas.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/Ameyanagi/IBR-XAS",
    version="0.1.0",
    zip_safe=False,
)
