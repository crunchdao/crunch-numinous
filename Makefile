PYTHON=python
PIP=$(PYTHON) -m pip

install:
	$(PIP) install -e .

uninstall:
	$(PIP) uninstall crunch-numinous -y

test:
	$(PYTHON) -m pytest -v

build:
	rm -rf dist
	$(PYTHON) -m build
	twine check dist/*

.PHONY: install uninstall test build
