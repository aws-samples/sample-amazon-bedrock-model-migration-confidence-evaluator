import pytest

from modelshift.sources import parse_s3_uri, S3UriError


def test_parse_valid_uri():
    assert parse_s3_uri("s3://my-bucket/logs/2026/") == ("my-bucket", "logs/2026/")
    assert parse_s3_uri("s3://my-bucket") == ("my-bucket", "")


@pytest.mark.parametrize("uri", [
    "https://my-bucket/logs",          # wrong scheme
    "s3:/my-bucket",                   # malformed
    "s3://",                           # no bucket
    "s3://UPPER/logs",                 # uppercase not allowed
    "s3://ab",                         # too short (<3)
    "s3://my..bucket/logs",            # consecutive dots
    "s3://10.0.0.1/logs",              # IP-style bucket
    "s3://bad_bucket/logs",            # underscore not allowed
    "s3://my-bucket/logs\nSET",        # control char
])
def test_parse_rejects_bad(uri):
    with pytest.raises(S3UriError):
        parse_s3_uri(uri)


def test_allowlist_enforced(monkeypatch):
    monkeypatch.setenv("MODELSHIFT_S3_ALLOWED_PREFIXES",
                       "s3://approved-bucket/logs/,s3://other-bucket/")
    # in the allowlist -> ok
    assert parse_s3_uri("s3://approved-bucket/logs/jan.jsonl") == ("approved-bucket", "logs/jan.jsonl")
    assert parse_s3_uri("s3://other-bucket/anything") == ("other-bucket", "anything")
    # valid bucket but NOT in allowlist -> rejected (SSRF guard)
    with pytest.raises(S3UriError):
        parse_s3_uri("s3://approved-bucket/secrets/creds")   # wrong prefix in an allowed bucket
    with pytest.raises(S3UriError):
        parse_s3_uri("s3://some-random-bucket/logs")         # bucket not allowed


def test_no_allowlist_permits_any_valid_bucket(monkeypatch):
    monkeypatch.delenv("MODELSHIFT_S3_ALLOWED_PREFIXES", raising=False)
    assert parse_s3_uri("s3://any-valid-bucket/x") == ("any-valid-bucket", "x")
