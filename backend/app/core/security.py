"""Authentication / authorization dependencies.

* `require_api_key` — header `X-API-Key`. Used for all read endpoints
  (predictions, monitoring read-outs).
* `require_admin` — JWT bearer token. Used for retrain, registry promotion,
  any state-changing admin action.
"""
from __future__ import annotations

import hmac
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from pydantic import BaseModel

from backend.app.core.config import Settings, get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)


class TokenPayload(BaseModel):
    sub: str
    role: str = "admin"
    exp: int


def require_api_key(
    api_key: Annotated[str | None, Depends(api_key_header)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> str:
    if api_key is None or not hmac.compare_digest(api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key


def create_access_token(subject: str, settings: Settings) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, object] = {
        "sub": subject,
        "role": "admin",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.jwt_expiry_minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def require_admin(
    token: Annotated[str | None, Depends(oauth2_scheme)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TokenPayload:
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        decoded = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    payload = TokenPayload(**decoded)
    if payload.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return payload


def authenticate_admin(username: str, password: str, settings: Settings) -> bool:
    """Constant-time credential check against env-configured admin user."""
    return (
        hmac.compare_digest(username, settings.admin_username)
        and hmac.compare_digest(password, settings.admin_password)
    )
