# 02460Miniprojects
Codebase for Miniprojects 1, 2 and 3 in course 02460 Advanced Machine Learning at the Technical University of Denmark.

## UV as package manager
Install guide (use standalone installer): https://docs.astral.sh/uv/getting-started/installation/

Each project contains a UV project for the packages that are required for that specific project. To run any code in a project, simply `uv sync --package <project1, project2, project3>` in the root or `uv sync --all-packages` to get all packages from all projects.

Add a package via `uv add --package <project1, project2, project3> <package_name>`.

Invoke a python file via `uv run python <file>.py`. (Not consistent with relative imports)

Alternatively invoke a python file using `uv run python -m <path>` (example: `uv run python -m Project1.src.vae.vae`).