# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitOverride=false, reportUnusedCallResult=false
# pyright: reportPrivateUsage=false, reportIncompatibleVariableOverride=false

from dataclasses import dataclass, field
from types import TracebackType
from typing import TypeAlias, cast
import asyncio
import sys
from fake_useragent import UserAgent
from aiohttp import (
    ClientResponseError,
    ClientSession,
    ClientTimeout,
    CookieJar,
    TCPConnector,
)
from yarl import URL

__all__ = (
    "Client",
    "Json",
    "JsonD",
)


Json: TypeAlias = dict[str, "Json"] | list["Json"] | str | int | float | bool | None


JsonD: TypeAlias = dict[str, Json]


@dataclass(slots=True)
class Client:
    session_key: str = field(repr=False)
    retries: int = 10
    min_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    session: ClientSession = field(init=False)

    def __post_init__(self):
        headers = {"User-Agent": UserAgent().chrome}

        jar = CookieJar()
        jar.update_cookies({"sessionKey": self.session_key}, URL("https://claude.ai/"))

        self.session = ClientSession(
            headers=headers,
            cookie_jar=jar,
            connector=TCPConnector(limit=0),
            timeout=ClientTimeout(total=None),
        )

    async def __aenter__(self):
        await self.session.__aenter__()

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        return await self.session.__aexit__(exc_type, exc_val, exc_tb)

    async def _refresh(self, path: str) -> Json:
        # i have never seen a 429 or in fact a 4xx error of any kind from this
        # api, nor ratelimit headers or fields on the returned stuff (i've seen
        # 403 Forbidden from cloudflare, in front of claude.ai, but only behind
        # vpn, i.e. totally blocked, and even then no ratelimit headers), so we
        # will have to cross our fingers and hope our rate and conn limiting is
        # enough not to break anything...
        async with self.session.get(
            f"https://claude.ai/api/{path}",
            allow_redirects=False,  # csrf defense for very silly threat model
        ) as r:
            # explicitly check r.status instead of using r.raise_for_status
            # because we want to raise for r.status >= 300 not just >= 400
            if 200 <= r.status < 300:
                return cast(Json, await r.json())

            raise ClientResponseError(
                r.request_info,
                r.history,
                status=r.status,
                message=r.reason or "",
                headers=r.headers,
            )

    async def refresh(self, path: str) -> Json:
        retry_delay = self.min_retry_delay
        for retry in range(self.retries - 1):
            try:
                return await self._refresh(path)
            except Exception as e:
                print(
                    f"Error fetching {path} (try {retry+1} of {self.retries}, "
                    + f"waiting {retry_delay:.1f}s): {e}",
                    file=sys.stderr,
                )
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)

        return await self._refresh(path)
