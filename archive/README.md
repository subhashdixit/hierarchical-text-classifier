# Original artifacts

Legacy pickle files and requirements were moved here unchanged during the Flask refactor. They are retained for provenance and are excluded from the Docker image. The deployed code never loads them. Original notebook: ../notebooks/original_exploration.ipynb.

Do not treat old model files as compatible with the new service. Joblib/pickle deserialization can execute code; only load trusted artifacts in a matched environment.
