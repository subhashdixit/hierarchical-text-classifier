"""Version and verify trusted, locally produced model artifacts."""

import hashlib
import json
from pathlib import Path
import joblib
import sklearn


def sha256(path: Path) -> str:
    """Hash file bytes without deserialization."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_bundle(bundle: dict, directory: Path, metadata: dict) -> None:
    """Write the model bundle and human-readable provenance."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "classifier.joblib"
    joblib.dump(bundle, path, compress=3)
    metadata = {
        **metadata,
        "model_sha256": sha256(path),
        "sklearn_version": sklearn.__version__,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_bundle(directory: Path) -> tuple[dict, dict]:
    """Check integrity/version before loading an approved artifact.

    Hashes are not signatures. Never accept uploaded joblib/pickle models:
    deserialization can execute code. This loader is only for repository assets.
    """
    metadata = json.loads((directory / "metadata.json").read_text())
    path = directory / "classifier.joblib"
    if sha256(path) != metadata["model_sha256"]:
        raise ValueError("Model checksum mismatch; restore an approved bundle.")
    if metadata["sklearn_version"] != sklearn.__version__:
        raise ValueError("Model requires the pinned scikit-learn version.")
    return joblib.load(path), metadata
