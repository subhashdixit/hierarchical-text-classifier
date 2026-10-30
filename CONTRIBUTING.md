# Contributing

Create a branch from main. Install requirements-dev.txt and the editable package. Run Black, Pylint, pytest and scripts/check_artifacts.py exactly as shown in the README. Build/run Docker for serving changes.

Open a pull request describing user-visible behaviour, validation and limitations. Keep training and test data isolated; never tune selection using final test labels. For model updates, include source/config, approved artifact, metadata and measured report together. Candidate training workflows do not promote automatically.

Do not commit .env, credentials, uploaded user text or sensitive employer material. Data rights must be clarified before introducing new datasets. Dependabot update PRs require the same tests; a green linter alone is not review.
