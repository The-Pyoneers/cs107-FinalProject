from setuptools import setup, find_packages

setup(
    name="farad",
    version="0.0.0",
    author="Harvard CS107 Final Project Group 34",
    description="A package for forward and reverse automatic differentiation",
    url="https://github.com/The-Pyoneers/cs107-FinalProject",
    tests_require=["pytest"],
    packages=['farad'],
    install_requires=['numpy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)