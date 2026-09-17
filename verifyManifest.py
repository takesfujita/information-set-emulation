"""Check the package manifest and its SHA-256 checksum (run from any directory)."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
manifest = root / "packageManifest.json"
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
expected = (root / "packageManifestSha256.txt").read_text().split()[0]
assert digest(manifest) == expected, "Manifest checksum mismatch"
entries = json.loads(manifest.read_text())["files"]
failures = []
for entry in entries:
    path = root / entry["path"]
    if not path.is_file() or path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
        failures.append(entry["path"])
if failures:
    raise SystemExit("MISMATCH: " + ", ".join(failures))
print(f"PASS: manifest and {len(entries)} files")
