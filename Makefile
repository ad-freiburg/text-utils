.PHONY: all
all: checkstyle test

.PHONY: checkstyle
checkstyle:
	flake8 python

.PHONY: test
test:
	cargo test
	pytest -W ignore

# preferred build command for local installation

.PHONY: build_native
build_native:
	pip install -r requirements.txt
	RUSTFLAGS="-C target-cpu=native" maturin build --release --compatibility linux
	pip install .
