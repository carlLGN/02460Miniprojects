# 02460Miniprojects
Codebase for Miniprojects 1, 2 and 3 in course 02460 Advanced Machine Learning at the Technical University of Denmark.

## UV as package manager
Install guide: https://docs.astral.sh/uv/getting-started/installation/

Each project contains a UV project for the packages that are required for that specific project. To run any code in a project, simply `uv sync` with the individual project as your root.

Add a package via `uv add <package>`.

Invoke a python file via `uv run python <file>.py`