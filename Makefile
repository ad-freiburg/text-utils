# preferred build command for local installation
.PHONY: build_native
build_native:
	maturin develop --release -- -C target-cpu=native
