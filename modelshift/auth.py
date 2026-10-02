"""Pluggable authentication for the ModelShift API.

Mode is selected by the MODELSHIFT_AUTH_MODE env var:

  none          - no auth (default; local/dev only — do NOT expose publicly).
  shared_secret - HTTP Basic against MODELSHIFT_AUTH_USER / MODELSHIFT_AUTH_PASSWORD
                  (constant-time compared). Self-contained; the CloudFormation
                  shared-secret variant generates the password in Secrets Manager
                  and injects both as task env.
  cognito_alb   - trust the identity headers an Application Load Balancer injects
                  after its authenticate-cognito action (x-amzn-oidc-identity /
                  x-amzn-oidc-data). Login is enforced at the ALB; the app only
                  admits requests that carry the ALB-signed identity header. Direct
                  (non-ALB) access has no header and is rejected.

Health/version endpoints stay open in every mode (needed for ELB/container health
checks); all other routes require auth.
"""
from __future__ import annotations

import base64
import hmac
import os
from typing import Optional

from fastapi import HTTPException, Request

# Paths that are always reachable without auth (health checks, build id).
OPEN_PATHS = {
    "/api/v1/health",
    "/api/v1/version",
    "/health",
}


def _mode() -> str:
    return (os.environ.get("MODELSHIFT_AUTH_MODE") or "none").strip().lower()


def _unauthorized(detail: str, headers: Optional[dict] = None) -> HTTPException:
    return HTTPException(status_code=401,
                         detail={"error": {"code": "unauthorized", "message": detail}},
                         headers=headers)


def _check_shared_secret(request: Request) -> None:
    expected_user = os.environ.get("MODELSHIFT_AUTH_USER") or ""
    expected_pass = os.environ.get("MODELSHIFT_AUTH_PASSWORD") or ""
    if not expected_user or not expected_pass:
        # Misconfigured: fail closed rather than allow everyone.
        raise _unauthorized("server auth not configured")
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("basic "):
        raise _unauthorized("basic auth required",
                            headers={"WWW-Authenticate": 'Basic realm="ModelShift"'})
    try:
        raw = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
        user, _, pwd = raw.partition(":")
    except Exception:  # noqa: BLE001 — malformed header -> reject
        raise _unauthorized("malformed credentials")
    # Constant-time compares to avoid user/password timing oracles.
    ok_user = hmac.compare_digest(user, expected_user)
    ok_pass = hmac.compare_digest(pwd, expected_pass)
    if not (ok_user and ok_pass):
        raise _unauthorized("invalid credentials",
                            headers={"WWW-Authenticate": 'Basic realm="ModelShift"'})


def _check_cognito_alb(request: Request) -> None:
    # The ALB's authenticate-cognito action injects these AFTER a successful login.
    # A caller reaching the app without them bypassed the ALB (or wasn't logged in).
    ident = request.headers.get("x-amzn-oidc-identity")
    data = request.headers.get("x-amzn-oidc-data")
    if not ident and not data:
        raise _unauthorized("cognito authentication required (no ALB identity header)")


def require_auth(request: Request) -> None:
    """FastAPI dependency: enforce the configured auth mode. Raises 401 on failure."""
    path = request.url.path
    if path in OPEN_PATHS:
        return
    mode = _mode()
    if mode == "none":
        return
    if mode == "shared_secret":
        _check_shared_secret(request)
        return
    if mode == "cognito_alb":
        _check_cognito_alb(request)
        return
    # Unknown mode -> fail closed.
    raise _unauthorized(f"unknown auth mode: {mode}")
