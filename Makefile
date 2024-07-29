ifeq ($(shell uname -s),Darwin)
    SED_INPLACE := sed -i ''
else
    SED_INPLACE := sed -i
endif

.PHONY: .check .clean-article .rebuild-article article archive


.check-commands:
	@command -v git >/dev/null 2>&1 || { echo "git is not installed. Aborting."; exit 1; }
	@command -v sed >/dev/null 2>&1 || { echo "sed is not installed. Aborting."; exit 1; }
	@command -v latexmk >/dev/null 2>&1 || { echo "latexmk is not installed. Aborting."; exit 1; }
	@command -v zip >/dev/null 2>&1 || { echo "zip is not installed. Aborting."; exit 1; }

# Target for building the LaTeX project
.build-article: .check-commands
	cd article && latexmk -pdf main.tex && cd -

# Target for cleaning the LaTeX build files
.clean-article: .check-commands
	@git clean -fdx ./article

.clean-archive: 
	@rm -f archive.zip

.install-deps: .venv/bin/aclpubcheck
	@poetry install

.finalize: article/main.tex
	$(SED_INPLACE) 's/\\usepackage\[review\]{coling}/\\usepackage{coling}/g' article/main.tex

clean: .clean-article .clean-archive

build: article archive

# Target for rebuilding the LaTeX project
article: .clean-article .build-article

check-article: .install-deps .finalize .clean-article .build-article
	@poetry run aclpubcheck --paper_type long article/main.pdf
	@git checkout article/main.tex

# Target for creating a zip archive
archive: .check .clean-archive .archiveignore pyproject.toml
	$(SED_INPLACE) 's/^\(authors =\).*/\1 ["anonymous"]/' pyproject.toml
	zip -r archive.zip . -x @.archiveignore
	git checkout pyproject.toml

