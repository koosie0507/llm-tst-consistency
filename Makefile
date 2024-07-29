ifeq ($(shell uname -s),Darwin)
    SED_INPLACE := sed -i ''
else
    SED_INPLACE := sed -i
endif

.PHONY: .check .clean-article .rebuild-article article archive


# Check for latexmk and zip presence
.check:
	@command -v git >/dev/null 2>&1 || { echo "git is not installed. Aborting."; exit 1; }
	@command -v sed >/dev/null 2>&1 || { echo "sed is not installed. Aborting."; exit 1; }
	@command -v latexmk >/dev/null 2>&1 || { echo "latexmk is not installed. Aborting."; exit 1; }
	@command -v zip >/dev/null 2>&1 || { echo "zip is not installed. Aborting."; exit 1; }

# Target for building the LaTeX project
.build-article: .check
	cd article && latexmk -pdf main.tex && cd -

# Target for cleaning the LaTeX build files
.clean-article: .check
	@git clean -fdx ./article

.clean-archive: 
	@rm -f archive.zip

clean: .clean-article .clean-archive

build: article archive

# Target for rebuilding the LaTeX project
article: .clean-article .build-article

# Target for creating a zip archive
archive: .check .clean-archive .archiveignore pyproject.toml
	$(SED_INPLACE) 's/^\(authors =\).*/\1 ["anonymous"]/' pyproject.toml
	zip -r archive.zip . -x @.archiveignore
	git checkout pyproject.toml

