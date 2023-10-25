# preferred build command for local installation
.PHONY: build_native
build_native:
	maturin develop --release -- -C target-cpu=native

.PHONY: format
format:
	cargo fmt --all

.PHONY: check
check:
	cargo clippy -- -D warnings

.PHONY: test
test:
	cargo test

.PHONY: bench
bench:
	cargo bench --all-features

.PHONY: all
all: format check test
