# pyright: reportAny=false, reportExplicitAny=false, reportUnusedCallResult=false

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
)
from dataclasses import dataclass, field
from itertools import chain, islice, zip_longest
from contextlib import suppress
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from types import TracebackType
from typing import Any, ClassVar, TypeAlias, TypedDict, TypeVar, cast
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
import browser_cookie3  # pyright: ignore[reportMissingTypeStubs]

__version__ = "0.1.5"

__all__ = ("__version__", "main", "Client", "Store")

Json: TypeAlias = dict[str, "Json"] | list["Json"] | str | int | float | bool | None

T = TypeVar("T")


def _cast(_typ: type[T], obj: Any) -> T:
    # cast without pyright complaining about non-overlapping types
    return cast(T, cast(object, obj))


class _APIObjectRequired(TypedDict):
    uuid: str


class APIObject(_APIObjectRequired, total=False):
    name: str


class Chat(APIObject):
    pass


class Organization(APIObject):
    capabilities: list[str]


class Membership(APIObject):
    organization: Organization


class Account(APIObject):
    memberships: list[Membership]


PathFragment: TypeAlias = APIObject | str

TTY = sys.stdout.isatty()


def uuid(obj: APIObject) -> str:
    return str(UUID(obj["uuid"]))  # round trip to throw on invalid uuid


def truncate(s: str, max_len: int) -> str:
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def get_terminal_width() -> int:
    if TTY:
        with suppress(AttributeError, OSError):
            return os.get_terminal_size().columns
    return 80


@dataclass(slots=True)
class Store:
    # migrate from unknown versions by not migrating anything, i.e. starting over
    MIGRATIONS: ClassVar[
        defaultdict[str | None, tuple[str, Callable[[str, str], None]]]
    ] = defaultdict(lambda: (__version__, lambda old, new: None))
    FILENAME_TRANS: ClassVar[dict[int, int]] = {
        ord(c): ord("_") for c in '<>:"|?*/\\ \t\n\r'
    }

    store_dir: Path
    force_refresh: bool = False

    def __post_init__(self):
        if not self.store_dir.exists():
            self.store_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            (self.store_dir / "version").write_text(__version__)
            return

        version = None
        with suppress(FileNotFoundError):
            version = (self.store_dir / "version").read_text().strip()

        if version == __version__:
            return

        with (
            # TemporaryDirectory uses 0700 perms, which we want for private chats
            TemporaryDirectory(
                prefix=f"{self.store_dir.name}-",
                dir=self.store_dir.parent,
            ) as old_store_dir,
            TemporaryDirectory(
                prefix=f"{self.store_dir.name}-",
                dir=self.store_dir.parent,
            ) as new_store_dir,
        ):
            shutil.copytree(self.store_dir, old_store_dir, dirs_exist_ok=True)

            while version != __version__:
                version, migrate_step = self.MIGRATIONS[version]
                migrate_step(old_store_dir, new_store_dir)
                old_store_dir, new_store_dir = new_store_dir, old_store_dir
                shutil.rmtree(new_store_dir)
                Path(new_store_dir).mkdir(mode=0o700)

            with TemporaryDirectory(
                prefix=f"{self.store_dir.name}-",
                dir=self.store_dir.parent,
            ) as adjacent_temp:
                shutil.copytree(old_store_dir, adjacent_temp, dirs_exist_ok=True)
                shutil.rmtree(self.store_dir)
                Path(adjacent_temp).rename(self.store_dir)

        (self.store_dir / "version").write_text(__version__)

    def _path(self, *path: PathFragment) -> Path:
        def _frag(p: PathFragment) -> str:
            if isinstance(p, str):
                return p
            else:
                name = p.get("name", "").translate(self.FILENAME_TRANS)
                return f"{name}-{uuid(p)}" if name else uuid(p)

        cache_file = self.store_dir / Path(*map(_frag, path)).with_suffix(".json")
        cache_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        return cache_file

    def save(self, data: Json, *path: PathFragment) -> None:
        if not path:
            return
        cache_file = self._path(*path)
        with NamedTemporaryFile(
            "w",
            prefix=f"{cache_file.name}-",
            dir=cache_file.parent,
        ) as f:
            json.dump(data, f)
            f.flush()
            Path(f.name).rename(cache_file)

    def load(self, *path: PathFragment) -> Json | None:
        if self.force_refresh:
            return None

        cache_file = self._path(*path)
        with suppress(FileNotFoundError, NotADirectoryError), cache_file.open() as f:
            return cast(Json, json.load(f))

    def rename(self, old: Iterable[PathFragment], new: Iterable[PathFragment]) -> None:
        self._path(*old).rename(self._path(*new))


@dataclass(slots=True)
class Client:
    store: Store
    session_key: str
    retries: int = 10
    min_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    success_delay: float = 0.1
    connections: int = 6
    session: ClientSession = field(init=False)
    total: int = field(init=False, default=0)
    seen: int = field(init=False, default=0)

    def __post_init__(self):
        headers = {"User-Agent": UserAgent().chrome}

        jar = CookieJar()
        jar.update_cookies({"sessionKey": self.session_key}, URL("https://claude.ai/"))

        self.session = ClientSession(headers=headers, cookie_jar=jar)

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

    async def gather(self, *awaitables: Awaitable[T]) -> list[T]:
        awaitables_list = list(awaitables)

        awaitables_iter = iter(awaitables_list)
        pending: set[asyncio.Task[T]] = set(
            map(asyncio.ensure_future, islice(awaitables_iter, self.connections))
        )

        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    await task

                    await asyncio.sleep(self.success_delay)

                    if next_task := next(awaitables_iter, None):
                        pending.add(asyncio.ensure_future(next_task))

            # gather all, in original order
            return await asyncio.gather(*awaitables_list)
        except:
            for task in pending:
                task.cancel()

            await asyncio.gather(*pending, return_exceptions=True)

            raise

    async def refresh(
        self, path: str, cache: bool | Iterable[PathFragment] = False
    ) -> Json:
        retry_delay = self.min_retry_delay
        r = None
        for _ in range(self.retries):
            r = await self.session.get(
                f"https://claude.ai/api/{path}", allow_redirects=False
            )
            if r.ok:
                data = cast(Json, await r.json())
                if cache is True:
                    self.store.save(data, path)
                elif cache:
                    self.store.save(data, *cache)
                return data
            else:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)

        r.raise_for_status()  # pyright: ignore[reportOptionalMemberAccess]
        raise ClientError  # pyright doesn't know above always throws. sad!

    def get_batch(
        self,
        batch: Iterable[Chat],
        organization: Organization,
        chats: dict[str, Chat],
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
                with suppress(FileNotFoundError):
                    self.store.rename((organization, old), (organization, chat))

            yield self.refresh(  # pyright: ignore[reportReturnType]
                f"organizations/{uuid(organization)}/chat_conversations/{uuid(chat)}"
                + f"?tree=True&rendering_mode=messages&render_all_tools=true",
                cache=(organization, chat),
            )

    async def get_organization_batches(
        self, organization: Organization, chats: dict[str, Chat]
    ) -> AsyncGenerator[Iterator[Awaitable[None]], None]:
        async for count, batch in self.get_paginated(
            f"organizations/{uuid(organization)}/chat_conversations", chats
        ):
            self.total += count
            yield self.get_batch(batch, organization, chats)

        # save in *reverse* chronological order (merely to match api)
        self.store.save(cast(Json, list(reversed(chats.values()))), organization)

    async def get_all_nth_batches(
        self,
    ) -> AsyncGenerator[Iterable[Awaitable[None]], None]:
        account = _cast(Account, await self.refresh("account"))
        self.total = self.seen = 0

        per_organization_batches: list[
            AsyncGenerator[Iterator[Awaitable[None]], None]
        ] = []

        for membership in account["memberships"]:
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
                for chat in _cast(list[Chat], self.store.load(organization) or ())
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

    async def get_batches(self) -> AsyncGenerator[Iterator[Awaitable[None]], None]:
        sentinel = object()
        async for nth_batches in self.get_all_nth_batches():
            # round robin across batches when yielded object is iterated over
            yield filter(None, chain.from_iterable(zip_longest(*nth_batches)))

    async def get_all(self) -> None:
        async for batch in self.get_batches():
            await self.gather(*batch)

    async def get_paginated(
        self,
        path: str,
        items: dict[str, Chat],
        offset: int = 0,
        limit: int = 20,
        key: Callable[[Chat], str] = uuid,
    ) -> AsyncGenerator[tuple[int, Iterator[Chat]], None]:
        # sliding window sync (chat_conversations is recently-modified-first)
        batch: dict[str, Chat] = {}

        if not items:
            for item in _cast(list[Chat], await self.refresh(path)):
                batch[key(item)] = item

            yield (len(batch), reversed(batch.values()))
            items.update(reversed(list(batch.items())))
            batch.clear()

        new_items = 0
        while True:
            page = _cast(
                list[Chat], await self.refresh(f"{path}?limit={limit}&offset={offset}")
            )
            item = None
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
            if len(page) >= limit and item and key(item) not in items:
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


@dataclass(slots=True)
class DefaultPath:
    path: str

    def __str__(self):  # pyright: ignore[reportImplicitOverride]
        try:
            return f"~/{Path(self.path).relative_to(Path.home())}"
        except ValueError:
            return self.path


def get_session_key() -> str:
    try:
        for cookie in browser_cookie3.load(domain_name=".claude.ai"):
            if cookie.name == "sessionKey" and cookie.value:
                return cookie.value
        raise RuntimeError("sessionKey cookie not found in browser")
    except Exception as e:
        raise RuntimeError(
            "Failed to load browser cookies. "
            + "Set CLAUDE_SESSION_KEY to your claude.ai sessionKey cookie."
        ) from e


async def _main() -> None:
    default: Callable[[str], Any] = lambda k: Client.__dataclass_fields__[k].default

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

    store = Store(store_dir=Path(args.backup_dir), force_refresh=args.force_refresh)

    session_key = os.environ.get("CLAUDE_SESSION_KEY") or get_session_key()
    async with Client(
        store=store,
        session_key=session_key,
        retries=args.retries,
        min_retry_delay=args.min_retry_delay,
        max_retry_delay=args.max_retry_delay,
        success_delay=args.success_delay,
        connections=args.connections,
    ) as client:
        await client.get_all()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
