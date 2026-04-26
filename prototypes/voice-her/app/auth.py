import hmac
import os
from base64 import b64decode
from dataclasses import dataclass

from fastapi import HTTPException, Request, WebSocket, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials


security = HTTPBasic(auto_error=False)


@dataclass(frozen=True)
class BasicAuthConfig:
    username: str
    password: str

    @classmethod
    def from_env(cls) -> "BasicAuthConfig":
        return cls(
            username=os.getenv("BASIC_AUTH_USERNAME", "arthur"),
            password=os.getenv("BASIC_AUTH_PASSWORD", ""),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.password)


def is_authorized(username: str, password: str, config: BasicAuthConfig | None = None) -> bool:
    config = config or BasicAuthConfig.from_env()
    if not config.enabled:
        return True
    return hmac.compare_digest(username, config.username) and hmac.compare_digest(password, config.password)


async def require_request_auth(request: Request) -> None:
    config = BasicAuthConfig.from_env()
    if not config.enabled:
        return

    credentials: HTTPBasicCredentials | None = await security(request)
    if credentials and is_authorized(credentials.username, credentials.password, config):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Basic"},
    )


async def require_websocket_auth(websocket: WebSocket) -> bool:
    config = BasicAuthConfig.from_env()
    if not config.enabled:
        return True

    header = websocket.headers.get("authorization", "")
    if not header.lower().startswith("basic "):
        return False

    try:
        decoded = b64decode(header.split(" ", 1)[1]).decode("utf-8")
        username, password = decoded.split(":", 1)
    except (ValueError, UnicodeDecodeError):
        return False

    return is_authorized(username, password, config)
