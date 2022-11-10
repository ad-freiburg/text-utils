.PHONY: all
all: checkstyle tests

.PHONY: checkstyle
checkstyle:
	flake8 src/text_correction_utils

.PHONY: tests
tests:
	pytest tests -n auto --disable-pytest-warnings
