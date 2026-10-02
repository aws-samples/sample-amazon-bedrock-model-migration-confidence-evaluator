"""Log source loaders: local file and S3 prefix (Doc 2 §2.1)."""
from __future__ import annotations

import os
import re
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
    yield p.read_bytes(), f"upload:{p.name}"


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
            if not is_log_key(key):
                skipped += 1
                continue
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            objects.append((body, f"s3://{bucket}/{key}"))
    objects.sort(key=lambda t: t[1])  # stable lexicographic order
    return objects, skipped
