"""Log source loaders: local file and S3 prefix (Doc 2 §2.1)."""
from __future__ import annotations

import io
import os
import re
import zipfile
from pathlib import Path
from typing import Iterator, List, Tuple

LOG_SUFFIXES = (".json", ".jsonl", ".json.gz", ".jsonl.gz", ".log", ".ndjson")

# S3 bucket naming rules (simplified but strict): 3-63 chars, lowercase letters,
# digits, hyphens, dots; must start/end alphanumeric; no consecutive dots; not an IP.
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9]$")
_IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


class S3UriError(ValueError):
    """Raised when an s3:// URI is malformed or not allowed."""


def _allowed_prefixes() -> List[str]:
    """Optional allowlist from MODELSHIFT_S3_ALLOWED_PREFIXES (comma-separated
    s3://bucket/prefix entries). Empty/unset = no allowlist restriction."""
    raw = os.environ.get("MODELSHIFT_S3_ALLOWED_PREFIXES", "")
    return [p.strip() for p in raw.split(",") if p.strip()]


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Validate and split an s3:// URI into (bucket, prefix). Raises S3UriError.

    Rejects: non-s3 schemes, missing/invalid bucket names, IP-style buckets,
    consecutive dots, and (if MODELSHIFT_S3_ALLOWED_PREFIXES is set) any URI not
    under an allowed bucket/prefix. This is the SSRF guard for the ingest path —
    the loader runs with the task role's credentials, so an unvalidated URI would
    let a caller read arbitrary S3 objects the role can reach.
    """
    if not isinstance(uri, str) or not uri.startswith("s3://"):
        raise S3UriError("S3 URI must start with s3://")
    rest = uri[len("s3://"):]
    if "\n" in rest or "\r" in rest:
        raise S3UriError("S3 URI contains control characters")
    bucket, _, prefix = rest.partition("/")
    if not bucket:
        raise S3UriError("S3 URI missing bucket")
    if ".." in bucket or _IP_RE.match(bucket) or not _BUCKET_RE.match(bucket):
        raise S3UriError(f"invalid S3 bucket name: {bucket!r}")
    allow = _allowed_prefixes()
    if allow:
        full = f"s3://{bucket}/{prefix}"
        if not any(full.startswith(a) for a in allow):
            raise S3UriError("S3 URI not in the allowed prefixes (MODELSHIFT_S3_ALLOWED_PREFIXES)")
    return bucket, prefix


def load_file(path: str) -> Iterator[Tuple[bytes, str]]:
    p = Path(path)
    data = p.read_bytes()
    if is_zip(data):
        objects, _skipped = load_zip_bytes(data, f"upload:{p.name}")
        yield from objects
        return
    yield data, f"upload:{p.name}"


# --------------------------------------------------------------------------- #
# Zip archives: a folder tree of log files consolidated into ONE run
# --------------------------------------------------------------------------- #
class ZipArchiveError(ValueError):
    """Raised when a zip is unreadable or exceeds the safety limits."""


def _zip_limits() -> Tuple[int, int]:
    """(max entries, max total uncompressed bytes); env-overridable."""
    max_entries = int(os.environ.get("MODELSHIFT_ZIP_MAX_ENTRIES", "50000"))
    max_mb = int(os.environ.get("MODELSHIFT_ZIP_MAX_UNCOMPRESSED_MB", "1024"))
    return max_entries, max_mb * 1024 * 1024


def is_zip(data: bytes) -> bool:
    """Sniff by magic bytes, not filename (local header or empty-archive marker)."""
    return data[:4] in (b"PK\x03\x04", b"PK\x05\x06")


def _is_junk_member(name: str) -> bool:
    """OS metadata that is never a log: macOS resource forks, dotfiles, Thumbs.db."""
    parts = [p for p in name.split("/") if p]
    if not parts:
        return True
    return parts[0] == "__MACOSX" or any(p.startswith(".") for p in parts) or parts[-1].lower() == "thumbs.db"


def load_zip_bytes(data: bytes, label: str) -> Tuple[List[Tuple[bytes, str]], int]:
    """Expand a zip (any folder depth) into (objects, skipped_non_log_count).

    Every log-suffixed member becomes one object, so a tree of per-request folders
    ingests as a single consolidated run. Read entirely in memory -- nothing is
    extracted to disk, so member paths can't escape a directory (no zip-slip).
    Windows-built archives store '\\' separators; they are normalised to '/'.
    Guards: entry-count and total-uncompressed caps (zip-bomb), declared sizes are
    re-checked against bytes actually read, encrypted members are skipped.
    Nested zips are not expanded (counted as skipped).
    """
    max_entries, max_bytes = _zip_limits()
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ZipArchiveError(f"not a readable zip archive: {e}") from e
    with zf:
        members = [i for i in zf.infolist() if not i.is_dir()]
        if len(members) > max_entries:
            raise ZipArchiveError(f"zip has {len(members)} files; limit is {max_entries}")
        declared = sum(i.file_size for i in members)
        if declared > max_bytes:
            raise ZipArchiveError(f"zip expands to {declared // (1024 * 1024)} MB; "
                                  f"limit is {max_bytes // (1024 * 1024)} MB")
        objects: List[Tuple[bytes, str]] = []
        skipped = 0
        total = 0
        for info in members:
            name = info.filename.replace("\\", "/").lstrip("/")
            if _is_junk_member(name):
                continue  # OS noise, not a user file: neither ingested nor counted
            if not is_log_key(name):
                skipped += 1
                continue
            if info.flag_bits & 0x1:  # encrypted
                skipped += 1
                continue
            with zf.open(info) as fh:
                # Read at most declared+1 bytes: a lying header can't inflate past the cap.
                body = fh.read(info.file_size + 1)
            if len(body) > info.file_size:
                raise ZipArchiveError(f"zip member {name!r} is larger than its declared size")
            total += len(body)
            if total > max_bytes:
                raise ZipArchiveError("zip expands beyond the uncompressed size limit")
            objects.append((body, f"{label}!{name}"))
    objects.sort(key=lambda t: t[1])  # stable lexicographic order across folders
    return objects, skipped


def is_log_key(key: str) -> bool:
    k = key.lower()
    return any(k.endswith(sfx) for sfx in LOG_SUFFIXES)


def load_s3(uri: str) -> Tuple[List[Tuple[bytes, str]], int]:
    """Load every log object under an s3://bucket/prefix (or a single key).

    Returns (objects, skipped_non_log_count). boto3 is imported lazily so the
    module stays importable without AWS deps for unit tests.
    """
    import boto3  # lazy

    bucket, prefix = parse_s3_uri(uri)
    s3 = boto3.client("s3")

    objects: List[Tuple[bytes, str]] = []
    skipped = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if key.lower().endswith(".zip"):
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                zobjs, zskipped = load_zip_bytes(body, f"s3://{bucket}/{key}")
                objects.extend(zobjs)
                skipped += zskipped
                continue
            if not is_log_key(key):
                skipped += 1
                continue
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            objects.append((body, f"s3://{bucket}/{key}"))
    objects.sort(key=lambda t: t[1])  # stable lexicographic order
    return objects, skipped
