# Variables
SPHINXOPTS    ?=
SPHINXBUILD   = python3 -m sphinx
SOURCEDIR     = source
BUILDDIR      = build
SCRIPTSDIR    = scripts
APIDIR        = source/api
GENERATEDDIR  = source/api/generated

.DEFAULT_GOAL := html

# Targets
.PHONY: all check-deps apidocs html view-html clean distclean confirm-distclean help

# Build all (API docs + HTML)
all: clean api-ref apidocs html

# Check dependencies
check-deps:
	@command -v python3 > /dev/null || { echo "Python3 not found. Aborting."; exit 1; }
	@python3 -c "import sphinx" 2>/dev/null || { echo "Sphinx not installed. Aborting."; exit 1; }

# Generate api_reference.py files for API documentation index
api-ref: check-deps
	@echo "Generating api_reference.py files for API documentation index..."
	python $(SCRIPTSDIR)/generate_api_reference.py

# Generate .rst files for API documentation
apidocs: check-deps
	@echo "Generating .rst files for API documentation..."
	sphinx-apidoc -o $(APIDIR) ../crispy --separate --templatedir=$(SOURCEDIR)/_templates
	python $(SCRIPTSDIR)/generate_api_index.py

# Build HTML documentation
html: check-deps
	@echo "Building HTML documentation..."
	$(SPHINXBUILD) -b html -j auto $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS)

# Open the built HTML documentation in a browser
view-html: html
	@echo "Opening built HTML documentation..."
	xdg-open $(BUILDDIR)/index.html || open $(BUILDDIR)/index.html || echo "Unable to open HTML documentation."

# Clean build and generated files
clean:
	@echo "Cleaning build and generated files..."
	rm -rf $(BUILDDIR) $(APIDIR)

# Confirm before performing a distclean
confirm-distclean:
	@read -p "Are you sure you want to perform a distclean? This will remove all generated files (y/n): " confirm; \
	if [ "$$confirm" != "y" ]; then \
		echo "Distclean aborted."; \
		exit 1; \
	fi

# Perform a deep clean (all generated files)
distclean: confirm-distclean clean
	@echo "Removing all auto-generated .rst files..."
	rm -f $(GENERATEDDIR)/*.rst
	rm -f $(SCRIPTSDIR)/api_reference.py

# Show help message
help:
	@echo "Makefile targets:"
	@echo "  make all                     Generate API docs and build HTML documentation."
	@echo "  make apidocs                 Generate .rst files for API documentation."
	@echo "  make html                    Build the HTML documentation."
	@echo "  make view-html               Open the built HTML documentation in a browser."
	@echo "  make clean                   Clean up build and generated files."
	@echo "  make distclean               Clean everything, including auto-generated files (requires confirmation)."
	@echo "  make help                    Show this help message."