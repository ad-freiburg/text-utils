from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as rdm:
    long_description = rdm.read()

with open("src/text_utils/version.py", "r", encoding="utf8") as vf:
    version = vf.readlines()[-1].strip().split()[-1].strip("\"'")

setup(
    name="text_utils",
    version=version,
    description="Text utilities for Deep NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Github": "https://github.com/bastiscode/text-utils",
    },
    author="Sebastian Walter",
    author_email="swalter@cs.uni-freiburg.de",
    python_requires=">=3.6",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    scripts=[
        "bin/wsc"
    ],
    install_requires=[
        "torch>=1.8.0",
        "einops>=0.3.0",
        "numpy>=1.19.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.49.0"
    ]
)
