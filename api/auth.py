"""Authentication + RBAC dependencies.

JWT bearer tokens carry `sub` (username), `role` and `uid`. `get_current_user`
validates the token; `require_role(...)` gates endpoints by role. Both raise the
shared tool-contract error envelope via the app's exception handler.
"""
from __future__ import annotations

from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.config import settings
from core.db import query_one
from core.security import create_access_token, verify_password

_bearer = HTTPBearer(auto_error=False)


def authenticate(username: str, password: str) -> dict[str, Any] | None:
    """Return the user row (without hash) if credentials are valid, else None."""
    row = query_one(
        "SELECT user_id, username, role, password_hash FROM users WHERE username=%s",
        (username,),
    )
    if not row:
        return None
    if not verify_password(password, row["password_hash"]):
        return None
    return {"user_id": row["user_id"], "username": row["username"], "role": row["role"]}


def issue_token(user: dict[str, Any]) -> str:
    return create_access_token(
        username=user["username"], role=user["role"], user_id=user["user_id"]
    )


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict[str, Any]:
    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(
            creds.credentials,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="token expired") from None
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="invalid token") from None
    return {
        "user_id": payload.get("uid"),
        "username": payload.get("sub"),
        "role": payload.get("role"),
    }


def require_role(*roles: str):
    """Dependency factory: allow only the given roles."""

    def _checker(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        if user.get("role") not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"requires role in {sorted(roles)}",
            )
        return user

    return _checker
