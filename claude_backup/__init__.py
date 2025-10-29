# pyright: reportExplicitAny=false, reportAny=false, reportUnusedCallResult=false

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Mapping,
)
from dataclasses import dataclass, field
from itertools import chain, islice, zip_longest
from contextlib import contextmanager, suppress
from tempfile import TemporaryDirectory
from typing import Any, Literal, TypeVar
from uuid import UUID
import asyncio
import json
import os
import shutil
import sys

from fake_useragent import UserAgent
from platformdirs import user_data_dir
from aiohttp import ClientError, ClientSession, CookieJar
from yarl import URL
import browser_cookie3

# TODO: compare file mtimes for updates instead of using json list (ehh)
# TODO: factor "cache" (which is really anything but) out of Client

__version__ = "0.1.4"

__all__ = ("__version__", "main", "Client")

TTY = sys.stdout.isatty()

# migrate from unknown versions by not migrating anything, i.e. starting over
MIGRATIONS = defaultdict(lambda: (__version__, lambda old, new: None))


@contextmanager
def umask(mask: int):
    old_mask = os.umask(mask)
    try:
        yield
    finally:
        _ = os.umask(old_mask)


def initialize_cache(backup_dir: str) -> None:
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir, exist_ok=True)
        with open(os.path.join(backup_dir, "version"), "w") as f:
            f.write(__version__)
        return

    version = None
    with suppress(FileNotFoundError), open(os.path.join(backup_dir, "version")) as f:
        version = f.read().strip()

    if version == __version__:
        return

    cp = lambda old, new: shutil.copytree(
        old,
        new,
        dirs_exist_ok=True,
        copy_function=shutil.copyfile,  # respects umask unlike copy/copy2
    )

    with (
        umask(0o0077),
        # TemporaryDirectory uses 0700 perms, which we want for private chats
        TemporaryDirectory(prefix="claude-backup-") as old_backup_dir,
        TemporaryDirectory(prefix="claude-backup-") as new_backup_dir,
    ):
        cp(backup_dir, old_backup_dir)

        while version != __version__:
            version, migrate_step = MIGRATIONS[version]
            migrate_step(old_backup_dir, new_backup_dir)
            old_backup_dir, new_backup_dir = new_backup_dir, old_backup_dir
            shutil.rmtree(new_backup_dir)
            os.makedirs(new_backup_dir)

        with TemporaryDirectory(
            prefix="claude-backup-", dir=os.path.dirname(backup_dir)
        ) as adjacent_temp:
            cp(old_backup_dir, adjacent_temp)
            shutil.rmtree(backup_dir)
            os.rename(adjacent_temp, backup_dir)

    with open(os.path.join(backup_dir, "version"), "w") as f:
        print(__version__, file=f)


def uuid(obj: Mapping[str, Any]) -> str:
    return str(UUID(obj["uuid"]))  # round trip to throw on invalid uuid


FILENAME_TRANS = {ord(c): ord("_") for c in '<>:"|?*/\\ \t\n\r'}


def cache(*parts: Mapping[str, Any]) -> str:
    name = lambda p: p.get("name", "").translate(FILENAME_TRANS)
    return os.path.join(
        *(f"{name(p)}-{uuid(p)}" if name(p) else uuid(p) for p in parts)
    )


def truncate(s: str, max_len: int) -> str:
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def get_terminal_width() -> int:
    if TTY:
        with suppress(AttributeError, OSError):
            return os.get_terminal_size().columns
    return 80


T = TypeVar("T")


def round_robin(*iterables: Iterable[T]) -> Iterator[T]:
    sentinel = object()
    yield from filter(  # pyright: ignore[reportReturnType]
        lambda x: x is not sentinel,
        chain.from_iterable(zip_longest(*iterables, fillvalue=sentinel)),
    )


async def gather_ratelimited(
    *awaitables: Awaitable[T],
    max_concurrent: int,
    delay: float = 0.0,
) -> list[T]:
    awaitables = list(awaitables)  # pyright: ignore[reportAssignmentType]

    awaitables_iter = iter(awaitables)
    pending = set(islice(awaitables_iter, max_concurrent))

    try:
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                await task

                await asyncio.sleep(delay)

                if next_task := next(awaitables_iter, None):
                    pending.add(next_task)  # pyright: ignore[reportArgumentType]

        # gather all, in original order
        return await asyncio.gather(*awaitables)
    except:
        for task in pending:
            task.cancel()  # pyright: ignore[reportAttributeAccessIssue]

        await asyncio.gather(*pending, return_exceptions=True)  # pyright: ignore[reportArgumentType]

        raise


@dataclass(slots=True)
class Client:
    backup_dir: str
    session_key: str
    retries: int = 10
    min_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    success_delay: float = 0.1
    connections: int = 6
    force_refresh: bool = False
    session: ClientSession = field(init=False)
    total: int = field(init=False, default=0)
    seen: int = field(init=False, default=0)

    def __post_init__(self):
        jar = CookieJar()
        jar.update_cookies(
            {"sessionKey": self.session_key},
            response_url=URL("https://claude.ai/"),
        )
        self.session = ClientSession(
            headers={"User-Agent": UserAgent().chrome}, cookie_jar=jar
        )

    async def __aenter__(self):
        await self.session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self.session.__aexit__(exc_type, exc_val, exc_tb)

    async def gather(self, *awaitables: Awaitable[T]) -> list[T]:
        return await gather_ratelimited(
            *awaitables,
            max_concurrent=self.connections,
            delay=self.success_delay,
        )

    def cache(
        self, cache: str | Literal[False], data: dict[Any, Any] | list[Any]
    ) -> None:
        if cache:
            cache_file = os.path.join(self.backup_dir, f"{cache}.json")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            temp_file = f"{cache_file}.new"
            with open(temp_file, "w") as f:
                json.dump(data, f)
            os.rename(temp_file, cache_file)

    def get_cached(
        self, cache: str | Literal[False]
    ) -> dict[Any, Any] | list[Any] | None:
        if cache and not self.force_refresh:
            cache_file = os.path.join(self.backup_dir, f"{cache}.json")
            with suppress(FileNotFoundError, NotADirectoryError), open(cache_file) as f:
                return json.load(f)

    async def refresh(
        self, path: str, cache: str | bool = False
    ) -> dict[Any, Any] | list[Any]:
        cache = path if cache is True else cache

        retry_delay = self.min_retry_delay
        r = None
        for _ in range(self.retries):
            r = await self.session.get(
                f"https://claude.ai/api/{path}", allow_redirects=False
            )
            if r.ok:
                data = await r.json()
                self.cache(cache, data)
                return data
            else:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)

        r.raise_for_status()  # pyright: ignore[reportOptionalMemberAccess]
        raise ClientError  # pyright doesn't know above always throws. sad!

    async def get(
        self, path: str, cache: str | bool = True
    ) -> dict[Any, Any] | list[Any]:
        cache = path if cache is True else cache
        return self.get_cached(cache) or await self.refresh(path, cache)

    def get_batch(
        self, batch, organization: dict[str, Any], chats: dict[str, dict[str, Any]]
    ) -> Iterator[Awaitable[None]]:
        for chat in batch:
            self.seen += 1

            progress = f"{self.seen:>{len(str(self.total))}}/{self.total} {uuid(chat)}"

            if name := chat.get("name"):
                max_name_len = max(10, get_terminal_width() - (len(progress) + 1))
                print(f"{progress} {truncate(name, max_name_len)}")
            else:
                print(progress)

            if old := chats.get(uuid(chat)):
                with suppress(FileNotFoundError, OSError):
                    os.rename(
                        os.path.join(
                            self.backup_dir, f"{cache(organization, old)}.json"
                        ),
                        os.path.join(
                            self.backup_dir, f"{cache(organization, chat)}.json"
                        ),
                    )

            yield self.refresh(  # pyright: ignore[reportReturnType]
                f"organizations/{uuid(organization)}/chat_conversations/{uuid(chat)}"
                + f"?tree=True&rendering_mode=messages&render_all_tools=true",
                cache(organization, chat),
            )

    async def get_organization_batches(
        self, organization: dict[str, Any], chats: dict[str, dict[str, Any]]
    ) -> AsyncGenerator[Iterator[Awaitable[None]], None]:
        async for count, batch in self.get_paginated(
            f"organizations/{uuid(organization)}/chat_conversations", chats
        ):
            self.total += count
            yield self.get_batch(batch, organization, chats)

        # save in *reverse* chronological order (merely to match api)
        self.cache(cache(organization), list(reversed(chats.values())))

    async def get_all_nth_batches(
        self,
    ) -> AsyncGenerator[Iterable[Awaitable[None]], None]:
        account = await self.get("account")
        self.total = self.seen = 0

        per_organization_batches = []

        for membership in account["memberships"]:  # pyright: ignore[reportCallIssue, reportArgumentType]
            organization = membership["organization"]

            if name := organization.get("name"):
                organization_name = f"{uuid(organization)} {name}"
            else:
                organization_name = f"{uuid(organization)}"

            if "chat" not in organization["capabilities"]:
                print(
                    f'Skipping organization {organization_name} without "chat" capability'
                )
                continue
            else:
                print(f"Fetching chats for organization {organization_name}")

            chats = {
                chat["uuid"]: chat
                for chat in self.get_cached(cache(organization)) or ()
            }
            self.total += len(chats)
            self.seen += len(chats)

            per_organization_batches.append(
                self.get_organization_batches(organization, chats)
            )

        while any(
            batches := await self.gather(
                *(
                    anext(organization_batches, iter(()))
                    for organization_batches in per_organization_batches
                )
            )
        ):
            yield batches  # pyright: ignore[reportReturnType]

    async def get_batches(self):
        async for nth_batches in self.get_all_nth_batches():
            yield round_robin(*nth_batches)  # pyright: ignore[reportArgumentType]

    async def get_all(self):
        async for batch in self.get_batches():
            await self.gather(*batch)

    async def get_paginated(
        self,
        path: str,
        items: dict[str, Any],
        offset: int = 0,
        limit: int = 20,
        key: Callable[[dict[str, Any]], str] = uuid,
    ) -> AsyncGenerator[tuple[int, Iterator[dict[str, Any]]], None]:
        # sliding window sync (chat_conversations is recently-modified-first)
        batch = {}

        if not items:
            for item in await self.refresh(path):
                batch[key(item)] = item

            yield (len(batch), reversed(batch.values()))
            items.update(reversed(list(batch.items())))
            batch.clear()

        new_items = 0
        while True:
            page = await self.refresh(f"{path}?limit={limit}&offset={offset}")
            for item in page:
                if batch_item := batch.get(key(item)):
                    # each time we see an item in two different pages, i.e. if
                    # `items` is always in reverse chronological order and we
                    # see an item in items[:n] AND items[n:] for some n, there
                    # is a new (or updated) item as items[0] we're now missing
                    # and should go get later when we're done with this batch
                    new_items += 1

                    if batch_item == item:
                        continue
                elif items.get(key(item)) == item:
                    # we've made it to stuff we'd already seen before this
                    # batch, after which point we expect to know every item
                    break

                batch[key(item)] = item

            # we *ought* to break from this loop by seeing something from a
            # prior batch, but just in case e.g. all items in `items` have
            # since been deleted (or moved/are counted in new_items), we're
            # also definitely done if they ran out of items for this page
            if (
                len(page) >= limit
                and key(item)  # pyright: ignore[reportPossiblyUnboundVariable]
                not in items
            ):
                offset += limit
                limit *= 2
                continue
            else:
                yield (len(batch), reversed(batch.values()))
                items.update(reversed(list(batch.items())))
                batch.clear()

                if new_items:
                    # go back for items added while we refreshed (plus one
                    # item that we expect to have seen, so if no more have
                    # been added we only need the one additional batch)
                    offset = 0
                    limit = new_items + 1
                    new_items = 0
                    continue

            break  # if we have neither next page nor new_items we are done


class DefaultPath:
    __slots__ = ("path",)

    def __init__(self, path: str):
        self.path = path

    def __str__(self) -> str:
        home = os.path.expanduser("~")
        if self.path.startswith(home):
            return "~" + self.path[len(home) :]
        return self.path


def get_session_key() -> str:
    try:
        jar = browser_cookie3.load(domain_name=".claude.ai")
        for cookie in jar:
            if cookie.name == "sessionKey" and cookie.value:
                return cookie.value
        raise RuntimeError("sessionKey cookie not found in browser")
    except Exception as e:
        raise RuntimeError(
            "Failed to load browser cookies. "
            + "Set CLAUDE_SESSION_KEY to your claude.ai sessionKey cookie."
        ) from e


async def _main() -> None:
    default = lambda k: Client.__dataclass_fields__[k].default

    parser = ArgumentParser(
        description="Backup Claude.ai chats",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "backup_dir",
        nargs="?",
        default=DefaultPath(user_data_dir("claude-backup")),
        help="Directory to save backups",
    )
    parser.add_argument(
        "-c",
        "--connections",
        type=int,
        default=default("connections"),
        help="Maximum concurrent connections",
    )
    parser.add_argument(
        "-s",
        "--success-delay",
        type=float,
        default=default("success_delay"),
        help="Delay after successful request in seconds",
    )
    parser.add_argument(
        "-r",
        "--retries",
        type=int,
        default=default("retries"),
        help="Number of retries for API requests",
    )
    parser.add_argument(
        "--min-retry-delay",
        type=float,
        default=default("min_retry_delay"),
        help="Minimum retry delay in seconds",
    )
    parser.add_argument(
        "--max-retry-delay",
        type=float,
        default=default("max_retry_delay"),
        help="Maximum retry delay in seconds",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-fetch all accounts and chats, ignoring cache",
    )

    args = parser.parse_args()
    if isinstance(args.backup_dir, DefaultPath):
        args.backup_dir = args.backup_dir.path

    initialize_cache(args.backup_dir)

    session_key = os.environ.get("CLAUDE_SESSION_KEY") or get_session_key()
    async with Client(session_key=session_key, **vars(args)) as client:
        await client.get_all()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
