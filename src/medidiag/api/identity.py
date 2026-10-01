"""服务器签发不可预测的匿名会话，客户端不能自报 owner。"""

from __future__ import annotations

import hashlib
import secrets
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool

from medidiag.db.models import WebSession

COOKIE_NAME = "medidiag_session"
SESSION_SECONDS = 30 * 24 * 3600


def identify(factory: sessionmaker[Session], token: str | None) -> tuple[str, str | None]:
    now = datetime.now(UTC).replace(tzinfo=None)
    with factory() as session:
        if token:
            if len(token) > 128:
                raise ValueError("无效会话")
            item = session.get(WebSession, hashlib.sha256(token.encode()).hexdigest())
            if item is None or item.expires_at <= now:
                raise ValueError("会话无效或已过期")
            return item.owner_id, None
        token = secrets.token_urlsafe(32)
        owner = "guest_" + uuid.uuid4().hex
        session.add(WebSession(token_hash=hashlib.sha256(token.encode()).hexdigest(),
                               owner_id=owner, expires_at=now + timedelta(seconds=SESSION_SECONDS)))
        session.commit()
        return owner, token


async def identity_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    if not request.url.path.startswith(("/api/", "/assistant", "/demo")):
        return await call_next(request)
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("origin")
        same_origin = str(request.url.replace(path="", query="", fragment="")).rstrip("/")
        if (origin and origin.rstrip("/") != same_origin) or request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"code": "ORIGIN_REJECTED", "detail": "拒绝跨站修改请求。"}, status_code=403)
    try:
        owner, new_token = await run_in_threadpool(
            identify, request.app.state.session_factory, request.cookies.get(COOKIE_NAME),
        )
    except ValueError:
        rejected = JSONResponse({"code": "SESSION_INVALID", "detail": "会话无效或已过期，请刷新后建立新的匿名会话。"}, status_code=401)
        rejected.delete_cookie(COOKIE_NAME)
        return rejected
    request.state.owner_id = owner
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Vary"] = "Cookie"
    if new_token:
        response.set_cookie(COOKIE_NAME, new_token, max_age=SESSION_SECONDS, httponly=True,
                            secure=request.url.scheme == "https", samesite="strict", path="/")
    return response
