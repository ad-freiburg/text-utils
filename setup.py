from setuptools import setup
from setuptools_rust import Binding, RustExtension

with open("README.md", "r", encoding="utf8") as rdm:
    long_description = rdm.read()

with open("text_correction_utils/version.py", "r", encoding="utf8") as vf:
    version = vf.readlines()[-1].strip().split()[-1].strip("\"'")

setup(
    name="text-correction-utils",
    version=version,
    description="Utilities for text correction tasks using Deep NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Github": "https://github.com/bastiscode/text-correction-utils",
    },
    author="Sebastian Walter",
    author_email="swalter@cs.uni-freiburg.de",
    python_requires=">=3.8",
    packages=["text_correction_utils"],
    install_requires=[
        "torch>=1.8.0",
        "einops>=0.3.0",
        "numpy>=1.19.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.49.0"
    ],
    extras_require={
        "test": [
            "pycodestyle>=2.8.0",
            "pytest>=6.2.0",
            "pytest-xdist>=2.5.0"
        ]
    },
    # add rust extensions here
    # just like c extensions they are not zip safe
    rust_extensions=[
        RustExtension(
            "text_correction_utils.",
            binding=Binding.PyO3
        )
    ],
    zip_safe=False
)
