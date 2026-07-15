"""Evidence integrity hashing."""
import hashlib


def compute_hashes(file_bytes: bytes) -> dict:
    """Compute SHA-256 and MD5 of the raw uploaded bytes.

    SHA-256 is the primary evidentiary hash (collision-resistant, standard
    in digital forensics). MD5 is included only as a secondary legacy
    reference since some existing case-management tools still expect it.
    """
    return {
        "sha256": hashlib.sha256(file_bytes).hexdigest(),
        "md5": hashlib.md5(file_bytes).hexdigest(),
        "size_bytes": len(file_bytes),
    }
