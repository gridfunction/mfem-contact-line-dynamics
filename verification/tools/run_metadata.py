"""Read completed run metadata against the verification case manifest."""

import json
from pathlib import Path


MANIFEST = Path(__file__).resolve().parents[1] / "run_manifest.json"


def load_completed_cases(runs, prefixes, allow_incomplete=False):
    entries = json.loads(MANIFEST.read_text())["verification_runs"]
    completed = {}
    missing = []
    for entry in entries:
        name = entry["case"]
        if not name.startswith(prefixes):
            continue
        descriptor = runs / name / "run.json"
        try:
            metadata = json.loads(descriptor.read_text())
        except FileNotFoundError:
            missing.append(name)
            continue
        if metadata.get("exit_code") != 0:
            missing.append(name)
            continue
        expected = {"name": name, **entry["settings"]}
        recorded = metadata["case"]
        if any(recorded.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Run metadata disagrees with the verification manifest: {descriptor}")
        completed[name] = recorded
    if missing and not allow_incomplete:
        raise ValueError("Missing successful runs: " + ", ".join(missing))
    return completed, missing
