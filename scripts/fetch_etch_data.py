"""Download the authors' 89.7 MB etching DFT archive and verify published MD5."""
import hashlib
from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parents[1]
path = root / "data/downloads/etch_dft.tar"
path.parent.mkdir(parents=True, exist_ok=True)
url = "https://zenodo.org/records/19491140/files/dft.tar?download=1"
if not path.exists():
    urllib.request.urlretrieve(url, path)
with path.open("rb") as stream:
    digest = hashlib.md5()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    checksum = digest.hexdigest()
if checksum != "c0b18d62a9ba8905b29fe3e90689c325":
    raise ValueError(f"Checksum mismatch: {checksum}")
print(path, checksum)
