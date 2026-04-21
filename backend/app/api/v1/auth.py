"""OAuth2 password-flow token endpoint for admin operations."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from backend.app.api.v1.schemas import TokenResponse
from backend.app.core.config import Settings, get_settings
from backend.app.core.security import authenticate_admin, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=TokenResponse, summary="Exchange admin credentials for a JWT")
def issue_token(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TokenResponse:
    if not authenticate_admin(form.username, form.password, settings):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(form.username, settings)
    return TokenResponse(access_token=token, expires_in=settings.jwt_expiry_minutes * 60)
