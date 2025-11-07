"""Setup configuration for Attention Is All You Need PyTorch implementation."""

from setuptools import setup, find_packages

with open("Readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="attention-is-all-you-need",
    version="0.1.0",
    author="Aya Nabil",
    description="PyTorch implementation of 'Attention Is All You Need' (Vaswani et al., 2017)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/1AyaNabil1/attention_is_all_you_need",
    packages=find_packages(exclude=["tests", "docs", "notebooks", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "transformer-train=src.train:main",
        ],
    },
)
