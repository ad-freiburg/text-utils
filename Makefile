.PHONY: all
all: checkstyle tests

.PHONY: checkstyle
checkstyle:
	flake8 text_correction_utils

.PHONY: tests
tests:
	pytest pytests -n auto --disable-pytest-warnings

# preferred build command for local installation

.PHONY: build_native
build_native:
	pip install -r requirements.txt
	RUSTFLAGS="-C target-cpu=native" maturin build --release --compatibility linux
	pip install .
