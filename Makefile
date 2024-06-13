.PHONY: .check .article .clean-article .rebuild-article .archive


# Check for latexmk and zip presence
.check:
	@command -v git >/dev/null 2>&1 || { echo "git is not installed. Aborting."; exit 1; }
	@command -v sed >/dev/null 2>&1 || { echo "sed is not installed. Aborting."; exit 1; }
	@command -v latexmk >/dev/null 2>&1 || { echo "latexmk is not installed. Aborting."; exit 1; }
	@command -v zip >/dev/null 2>&1 || { echo "zip is not installed. Aborting."; exit 1; }

# Target for building the LaTeX project
.article: .check
	cd article && latexmk -pdf main.tex && cd -

# Target for cleaning the LaTeX build files
.clean-article: .check
	cd article && latexmk -c main.tex && cd -

# Target for rebuilding the LaTeX project
.rebuild-article: .clean-article .article

# Target for creating a zip archive
.archive: .check .archiveignore pyproject.toml
	@rm -f archive.zip
	sed -i '' 's/^\(authors =\).*/\1 ["anonymous"]/' pyproject.toml
	zip -r archive.zip . -x @.archiveignore
	git checkout pyproject.toml

build: .rebuild-article .archive