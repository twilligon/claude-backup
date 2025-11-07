"""Unofficial, unsanctioned tool to backup Claude.ai chats to local files."""

# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitOverride=false, reportUnusedCallResult=false
# pyright: reportPrivateUsage=false, reportIncompatibleVariableOverride=false

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from asyncio import Task
from collections import defaultdict
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
)
from dataclasses import dataclass, field
from contextlib import aclosing, suppress
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from types import TracebackType
from typing import Any, ClassVar, TypeAlias, TypeVar, cast, final
from uuid import UUID
import asyncio
import json
import os
import shutil
import sys

from fake_useragent import UserAgent
from platformdirs import user_data_dir
from aiohttp import ClientError, ClientSession, CookieJar, TCPConnector
from yarl import URL
import browser_cookie3  # pyright: ignore[reportMissingTypeStubs]

__version__ = "0.1.8"

__all__ = (
    "__version__",
    "main",
    "get_session_key",
    "Client",
    "Store",
    "Syncer",
    "APIObject",
    "Immutable",
    "Nameable",
    "Loadable",
    "Account",
    "Membership",
    "Organization",
    "Chats",
    "ChatsEntry",
    "Chat",
    "Json",
    "JsonD",
)

Json: TypeAlias = dict[str, "Json"] | list["Json"] | str | int | float | bool | None
JsonD: TypeAlias = dict[str, Json]

T = TypeVar("T")
T_APIObject = TypeVar("T_APIObject", bound="APIObject")


@dataclass(slots=True)
class Client:
    session_key: str
    retries: int = 10
    min_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    session: ClientSession = field(init=False)

    def __post_init__(self):
        headers = {"User-Agent": UserAgent().chrome}

        jar = CookieJar()
        jar.update_cookies({"sessionKey": self.session_key}, URL("https://claude.ai/"))

        self.session = ClientSession(
            headers=headers, cookie_jar=jar, connector=TCPConnector(limit=0)
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

    async def refresh(self, path: str) -> Json:
        retry_delay = self.min_retry_delay
        r = None
        for _ in range(self.retries):
            # i have never seen a 429 or in fact a 4xx error of any kind from
            # this api, nor ratelimit headers or fields in the returned stuff
            # so we will have to cross our fingers and hope our rate and conn
            # limiting is enough not to break anything...
            async with self.session.get(
                f"https://claude.ai/api/{path}", allow_redirects=False
            ) as r:
                if r.status in range(200, 300):
                    data = cast(Json, await r.json())
                    return data
                elif r.status in range(300, 400):
                    raise ClientError(
                        f"Unexpected redirect (HTTP {r.status}) - "
                        + "session key may be invalid or API endpoint changed"
                    )
                else:
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, self.max_retry_delay)

        assert r is not None  # never happens unless self.retries <= 0
        r.raise_for_status()
        raise ClientError  # pyright doesn't know above always throws. sad!


@dataclass(slots=True)
class Store:
    @staticmethod
    def _cp(old: str | Path, new: str | Path) -> None:
        shutil.copytree(old, new, dirs_exist_ok=True)

    MIGRATIONS: ClassVar[
        defaultdict[str | None, tuple[str, Callable[[str, str], None]]]
    ] = defaultdict(
        # migrate from unknown versions by not migrating anything/starting over
        lambda: (__version__, lambda old, new: None),
        {"0.1.7": ("0.1.8", _cp)},  # if format unchanged, migrate whole store
    )

    store_dir: Path
    ignore_cache: bool = False

    def __post_init__(self):
        if not self.store_dir.exists():
            self.store_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            (self.store_dir / "version").write_text(f"{__version__}\n")
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
            self._cp(self.store_dir, old_store_dir)

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
                self._cp(old_store_dir, adjacent_temp)
                shutil.rmtree(self.store_dir)
                Path(adjacent_temp).rename(self.store_dir)

        (self.store_dir / "version").write_text(f"{__version__}\n")

    def save(self, path: Path, data: Json) -> None:
        cache_file = self.store_dir / path.with_suffix(".json")
        cache_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

        with NamedTemporaryFile(
            "w",
            prefix=f"{cache_file.name}-",
            dir=cache_file.parent,
        ) as f:
            json.dump(
                data, f, ensure_ascii=False, check_circular=False, separators=(",", ":")
            )
            f.flush()
            Path(f.name).rename(cache_file)

    def load(self, path: Path) -> Json | None:
        if self.ignore_cache:
            return None

        cache_file = self.store_dir / path.with_suffix(".json")
        with suppress(FileNotFoundError, NotADirectoryError), cache_file.open() as f:
            return cast(Json, json.load(f))

    def delete(self, path: Path) -> None:
        file = self.store_dir / path.with_suffix(".json")
        with suppress(FileNotFoundError):
            file.unlink()

        parent = file.parent
        while parent != self.store_dir and parent.exists():
            try:
                parent.rmdir()
                parent = parent.parent
            except OSError:
                break


class APIObject:
    _client: Client  # pyright: ignore[reportUninitializedInstanceVariable]
    _store: Store  # pyright: ignore[reportUninitializedInstanceVariable]
    _data: Json  # pyright: ignore[reportUninitializedInstanceVariable]

    @property
    def client(self) -> Client:
        return self._client

    @property
    def store(self) -> Store:
        return self._store

    def get_data(self) -> Json:
        return self._data

    def set_data(self: T_APIObject, data: Json) -> T_APIObject:
        self._data = data
        return self

    def api_path(self) -> str:
        raise NotImplementedError

    def store_path(self) -> Path:
        raise NotImplementedError

    @classmethod
    def _load(
        cls: type[T_APIObject],
        *args: Any,
        store_path: Path | None = None,
    ) -> T_APIObject | None:
        obj = cls(*args)
        if data := obj.store.load(store_path or obj.store_path()):
            return obj.set_data(data)
        else:
            return None

    @classmethod
    async def _fetch(
        cls: type[T_APIObject],
        *args: Any,
        api_path: str | None = None,
    ) -> T_APIObject:
        obj = cls(*args)
        data = await obj.client.refresh(api_path or obj.api_path())
        return obj.set_data(data).save()

    async def refresh(self: T_APIObject) -> T_APIObject:
        return self.set_data(await self.client.refresh(self.api_path())).save()

    def save(self: T_APIObject) -> T_APIObject:
        self.store.save(self.store_path(), self.get_data())
        return self

    def delete_cached(self) -> None:
        self.store.delete(self.store_path())


class Immutable(APIObject):
    def __hash__(self) -> int:
        return hash(
            json.dumps(
                self._data,
                ensure_ascii=False,
                sort_keys=True,
                check_circular=False,
                separators=(",", ":"),
            )
        )

    def __eq__(self, other: object) -> bool:
        if type(self) is type(other):
            assert isinstance(other, APIObject)  # appease pyright
            return self._data == other._data
        else:
            return NotImplemented


class Nameable(APIObject):
    _data: JsonD

    FILENAME_XLAT: ClassVar[dict[int, int]] = {
        ord(c): ord("_") for c in '<>:"|?*/\\ \t\n\r'
    }

    @property
    def uuid(self) -> str:
        return str(UUID(cast(str, self._data["uuid"])))

    @property
    def name(self) -> str | None:
        # api seems to use {"name": ""} to mean {"name": null}
        return cast(str, self._data.get("name")) or None

    def slug(self) -> str:
        if self.name:
            return f"{self.name.translate(self.FILENAME_XLAT)}-{self.uuid}"
        return self.uuid

    def __str__(self) -> str:
        if self.name:
            return f"{self.name} ({self.uuid})"
        return self.uuid


T_Loadable = TypeVar("T_Loadable", bound="Loadable")


class Loadable(APIObject):
    @classmethod
    def load(cls: type[T_Loadable], client: Client, store: Store) -> T_Loadable | None:
        return cls._load(client, store)

    @classmethod
    async def fetch(cls: type[T_Loadable], client: Client, store: Store) -> T_Loadable:
        return await cls._fetch(client, store)


@final
class Chat(Nameable):
    __slots__ = (
        "chat_list",
        "_data",
    )

    chat_list: "Chats"
    _data: JsonD

    def __init__(
        self,
        chat_list: "Chats",
    ):
        self.chat_list = chat_list

    @property
    def client(self) -> Client:
        return self.chat_list.client

    @property
    def store(self) -> Store:
        return self.chat_list.store

    def api_path(self) -> str:
        return (
            f"{self.chat_list.api_path()}/{self.uuid}"
            + "?tree=True&rendering_mode=messages&render_all_tools=true"
        )

    def store_path(self) -> Path:
        return self.chat_list.store_path() / self.slug()


@final
class ChatsEntry(Nameable, Immutable):
    __slots__ = (
        "chat_list",
        "_data",
    )

    chat_list: "Chats"
    _data: JsonD

    def __init__(
        self,
        chat_list: "Chats",
    ):
        self.chat_list = chat_list

    @property
    def client(self) -> Client:
        return self.chat_list.client

    @property
    def store(self) -> Store:
        return self.chat_list.store

    def chat_api_path(self) -> str:
        return (
            f"{self.chat_list.api_path()}/{self.uuid}"
            + "?tree=True&rendering_mode=messages&render_all_tools=true"
        )

    def chat_store_path(self) -> Path:
        return self.chat_list.store_path() / self.slug()

    def load_chat(self) -> Chat | None:
        return Chat._load(self.chat_list, store_path=self.chat_store_path())

    async def fetch_chat(self) -> Chat:
        return await Chat._fetch(self.chat_list, api_path=self.chat_api_path())


@final
class Chats(APIObject):
    __slots__ = ("organization", "unseen", "_data")

    organization: "Organization"
    _data: dict[str, JsonD]
    unseen: int

    def __init__(self, organization: "Organization"):
        self.organization = organization
        self.unseen = 0
        self._data = {}

    @property
    def client(self) -> Client:
        return self.organization.client

    @property
    def store(self) -> Store:
        return self.organization.store

    def api_path(self) -> str:
        return f"{self.organization.api_path()}/chat_conversations"

    def store_path(self) -> Path:
        return self.organization.store_path() / "chat_conversations"

    def set_data(self, data: Json) -> "Chats":
        # convert list (reverse chronological from API) to dict (forward chronological)
        self._data = {
            cast(str, entry["uuid"]): entry
            for entry in reversed(cast(list[JsonD], data))
        }
        return self

    @classmethod
    def _load(
        cls,
        *args: Any,
        store_path: Path | None = None,
    ) -> "Chats | None":
        obj = cls(*args)
        if data := obj.store.load(store_path or obj.store_path()):
            return obj.set_data(data)
        else:
            return None

    def get_data(self) -> Json:
        # convert dict (forward chronological) back to list (reverse chronological)
        return list(reversed(self._data.values()))

    def entry(self, uuid: str) -> ChatsEntry | None:
        if entry := self._data.get(uuid):
            return ChatsEntry(self).set_data(entry)
        else:
            return None

    def cached_entries(self) -> Iterator[ChatsEntry]:
        # yield in reverse chronological order (newest first)
        for entry in reversed(self._data.values()):
            yield ChatsEntry(self).set_data(entry)

    async def new_entries(
        self, page_size: int = 20, save: bool = True
    ) -> AsyncGenerator[ChatsEntry, None]:
        # if no data/first fetch, fetch all entries---they're all new
        if not self._data:
            self.set_data(await self.client.refresh(self.api_path()))

            for entry in self.cached_entries():
                yield entry

            if save:
                self.save()
            return

        # "sliding window" sync (chat_conversations is recently-modified-first)
        new: dict[str, JsonD] = {}

        offset = 0
        limit = self.unseen + 1 if self.unseen else page_size
        self.unseen = 0

        assert limit

        while True:
            page = cast(
                list[JsonD],
                await self.client.refresh(
                    f"{self.api_path()}?limit={limit}&offset={offset}"
                ),
            )

            done = False
            for entry in page:
                uuid = cast(str, entry["uuid"])

                if new_entry := new.get(uuid):
                    # this api doesn't have cursors or snapshots or anything :(
                    # so if an entry's created or updated *between our fetching
                    # one page and the next*, there's a *new* most recent entry
                    # which bumps all the others down and causes the last entry
                    # of the prior page also to be the first entry of the next:
                    #
                    # page 1 sees [A B]:  [A B] C D E
                    # entry D is updated: D A B C E
                    # page 2 sees [B C]:  D A [B C] E
                    #
                    # of course this can happen multiple times, and in fact the
                    # number of times it happens is the count of new entries we
                    # would expect at offsets 0 through N on our next fetch! so
                    # update self.unseen as a hint to the next new_entries call
                    # that there are likely exactly self.unseen new or changed.
                    self.unseen += 1

                    # ...but if the above hypothesis is wrong for some perverse
                    # reason like all the chats up until now being rewritten in
                    # a specific order behind our backs, still yield the entry:
                    #
                    # page 1 sees [A B]:  [A B] C D E
                    # B, D, & A updated:  A' D' B' C E
                    # page 2 sees [B' C]: A' D' [B' C] E
                    #
                    # so in case the cartesian daemon of claude chats is out to
                    # get us, we need to yield B' even if we wouldn't yield B.
                    if entry == new_entry:
                        continue
                elif entry == self._data.get(uuid):
                    # entry was in a chronological list of ones we already had,
                    # so we must also have all entries before it, so we're done
                    done = True
                    break

                new[uuid] = entry
                yield ChatsEntry(self).set_data(entry)

            # we *ought* to break from this loop by seeing something from prior
            # refreshes, but just in case e.g. all seen entries were deleted on
            # claude.ai (or more likely moved/counted in self.unseen)... we are
            # also definitely done if they ran out of items for this page
            if done or len(page) < limit:
                break

            # if not, double it and give it to the next person
            offset += limit
            limit *= 2

        # dicts preserve order, but because there is no dict.prepend(), we need
        # self._data to be in "forward-chronological" order so newly discovered
        # chats belong at the "end" w/r/t iteration order. thus we insert items
        # from necessarily reverse-chronological new in reverse so self._data's
        # still entirely in chronological order. this may be a bit galaxy brain
        self._data.update(reversed(new.items()))
        if save:
            self.save()

    async def refresh(self) -> "Chats":
        async for _ in self.new_entries():
            pass

        return self

    def __len__(self) -> int:
        return len(self._data)


@final
class Organization(Nameable, Loadable):
    __slots__ = (
        "_client",
        "_store",
        "_data",
    )

    _client: Client
    _store: Store
    _data: JsonD

    def __init__(self, client: Client, store: Store):
        self._client = client
        self._store = store

    def api_path(self) -> str:
        return f"organizations/{self.uuid}"

    def store_path(self) -> Path:
        return Path("organizations") / self.slug()

    @property
    def capabilities(self) -> list[str]:
        return cast(list[str], self._data["capabilities"])

    def chat_list(self) -> Chats:
        return Chats._load(self) or Chats(self)


@final
class Membership(Immutable):
    __slots__ = (
        "account",
        "_data",
    )

    account: "Account"
    _data: JsonD

    def __init__(self, account: "Account"):
        self.account = account

    @property
    def client(self) -> Client:
        return self.account.client

    @property
    def store(self) -> Store:
        return self.account.store

    def organization(self) -> Organization:
        return Organization(self.client, self.store).set_data(
            cast(JsonD, self._data["organization"])
        )


@final
class Account(Nameable, Loadable):
    __slots__ = (
        "_client",
        "_store",
        "_data",
    )

    _client: Client
    _store: Store
    _data: JsonD

    def __init__(self, client: Client, store: Store):
        self._client = client
        self._store = store

    def api_path(self) -> str:
        return "account"

    def store_path(self) -> Path:
        return Path("account")

    def memberships(self) -> Iterator[Membership]:
        for membership in cast(list[JsonD], self._data["memberships"]):
            yield Membership(self).set_data(membership)

    def organization(self, uuid: str) -> Organization | None:
        for membership in self.memberships():
            organization = membership.organization()
            if organization.uuid == uuid:
                return organization

        return None


def truncate(s: str, max_len: int) -> str:
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


async def aroundrobin(*iterators: AsyncGenerator[T, None]) -> AsyncGenerator[T, None]:
    try:
        done = False
        while not done:
            done = True
            for it in iterators:
                with suppress(StopAsyncIteration):
                    yield await anext(it)
                    done = False
    finally:
        for it in iterators:
            with suppress(BaseException):
                await it.aclose()


@dataclass(slots=True)
class Syncer:
    client: Client
    store: Store
    connections: int = 6
    success_delay: float = 0.1
    tty: bool = field(default_factory=sys.stdout.isatty)

    async def _as_completed(
        self, awaitables: AsyncIterator[Awaitable[T]]
    ) -> AsyncGenerator[Task[T], None]:
        pending: set[Task[T]] = set()

        try:
            for _ in range(self.connections):
                try:
                    awaitable = await anext(awaitables)
                    pending.add(asyncio.ensure_future(awaitable))
                except StopAsyncIteration:
                    break

            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    yield task

                    await asyncio.sleep(self.success_delay)

                    try:
                        awaitable = await anext(awaitables)
                        pending.add(asyncio.ensure_future(awaitable))
                    except StopAsyncIteration:
                        pass
        except BaseException:
            for task in pending:
                task.cancel()
            for task in pending:
                with suppress(BaseException):
                    await task
            raise
        finally:
            if isinstance(awaitables, AsyncGenerator):
                await awaitables.aclose()

    def as_completed(
        self, awaitables: Iterable[Awaitable[T]] | AsyncIterator[Awaitable[T]]
    ) -> AsyncGenerator[Task[T], None]:
        async def asyncify(
            iterable: Iterable[Awaitable[T]],
        ) -> AsyncGenerator[Awaitable[T], None]:
            for item in iterable:
                yield item

        if isinstance(awaitables, AsyncIterator):
            return self._as_completed(awaitables)
        else:
            return self._as_completed(asyncify(awaitables))

    async def gather(self, *awaitables: Awaitable[T]) -> list[T]:
        tasks_list = [asyncio.ensure_future(a) for a in awaitables]

        async with aclosing(self.as_completed(tasks_list)) as gen:
            async for task in gen:
                await task

        return [task.result() for task in tasks_list]

    async def get_organizations(self) -> AsyncGenerator[Organization]:
        old_account = Account.load(self.client, self.store)
        account = await Account.fetch(self.client, self.store)

        for membership in account.memberships():
            organization = membership.organization()

            if (
                old_account
                and (old_organization := old_account.organization(organization.uuid))
                and old_organization.store_path() != organization.store_path()
            ):
                print(
                    f"Renaming organization {old_organization} to {organization.name or organization.uuid}",
                    file=sys.stderr,
                )
                old_dir = self.store.store_dir / old_organization.store_path()
                new_dir = self.store.store_dir / organization.store_path()
                with suppress(FileNotFoundError):
                    old_dir.rename(new_dir)

            if "chat" not in organization.capabilities:
                print(
                    f'Skipping organization {organization} without "chat" capability',
                    file=sys.stderr,
                )
                continue

            print(f"Fetching chats for organization {organization}", file=sys.stderr)
            yield organization

    def print_entry(self, entry: ChatsEntry) -> None:
        name = entry.name or ""
        if self.tty:
            try:
                width = os.get_terminal_size().columns
            except OSError:
                width = 80
            name = truncate(name, width - 36 - 4)
        print(f"{entry.uuid}\t{name}")

    @staticmethod
    def fetch_new_chat(entry: ChatsEntry) -> Task[Chat]:
        # we must get old_entry *now* and not in the async function below since
        # Chats.new_entries might finish and save the new entry over old_entry!
        old_entry = entry.chat_list.entry(entry.uuid)
        old_chat = old_entry.load_chat() if old_entry else None

        async def _fetch_new_chat() -> Chat:
            chat = await entry.fetch_chat()

            if old_chat and old_chat.store_path() != chat.store_path():
                old_chat.delete_cached()

            return chat

        return asyncio.create_task(_fetch_new_chat())

    async def new_chat_fetches(self) -> AsyncGenerator[Task[Chat], None]:
        needs_refresh: list[tuple[Chats, list[Task[Chat]], Task[None]]] = [
            (organization.chat_list(), [], asyncio.create_task(asyncio.sleep(0)))
            async for organization in self.get_organizations()
        ]

        try:
            while needs_refresh:
                async with aclosing(
                    aroundrobin(
                        *(
                            (
                                (entry, fetches)
                                async for entry in chats.new_entries(save=False)
                            )
                            for chats, fetches, _ in needs_refresh
                        )
                    )
                ) as items:
                    async for entry, fetches in items:
                        fetch = self.fetch_new_chat(entry)
                        fetches.append(fetch)
                        self.print_entry(entry)
                        yield fetch

                for _, _, old_save in needs_refresh:
                    if old_save:
                        old_save.cancel()

                async def save_after_fetches(
                    chats: Chats, fetches: list[Task[Chat]]
                ) -> None:
                    for fetch in fetches:
                        await fetch

                    chats.save()

                needs_refresh = [
                    (
                        chats,
                        [],
                        asyncio.create_task(save_after_fetches(chats, fetches)),
                    )
                    for chats, fetches, _ in needs_refresh
                    if chats.unseen
                ]

            for _, _, save in needs_refresh:
                await save
        except BaseException:
            for _, _, save in needs_refresh:
                save.cancel()

                with suppress(BaseException):
                    await save

            raise

    async def sync_all(self) -> None:
        async with aclosing(self.as_completed(self.new_chat_fetches())) as tasks:
            async for task in tasks:
                await task


@dataclass(slots=True)
class DefaultPath:
    path: str

    def __str__(self):
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
    def default(cls: type[Any], key: str) -> Any:
        return cls.__dataclass_fields__[key].default

    parser = ArgumentParser(
        description="Backup Claude.ai chats",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
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
        default=default(Syncer, "connections"),
        help="Maximum concurrent connections",
    )
    parser.add_argument(
        "-d",
        "--success-delay",
        type=float,
        metavar="DELAY",
        default=default(Syncer, "success_delay"),
        help="Delay after successful request in seconds",
    )
    parser.add_argument(
        "-r",
        "--retries",
        type=int,
        default=default(Client, "retries"),
        help="Number of retries for API requests",
    )
    parser.add_argument(
        "--min-retry-delay",
        type=float,
        metavar="DELAY",
        default=default(Client, "min_retry_delay"),
        help="Minimum retry delay in seconds",
    )
    parser.add_argument(
        "--max-retry-delay",
        type=float,
        metavar="DELAY",
        default=default(Client, "max_retry_delay"),
        help="Maximum retry delay in seconds",
    )
    parser.add_argument(
        "--ignore-cache",
        action="store_true",
        help="Ignore local cache and re-fetch everything from API",
    )

    args = parser.parse_args()
    if isinstance(args.backup_dir, DefaultPath):
        args.backup_dir = args.backup_dir.path

    store = Store(store_dir=Path(args.backup_dir), ignore_cache=args.ignore_cache)

    session_key = os.environ.get("CLAUDE_SESSION_KEY") or get_session_key()
    async with Client(
        session_key=session_key,
        retries=args.retries,
        min_retry_delay=args.min_retry_delay,
        max_retry_delay=args.max_retry_delay,
    ) as client:
        syncer = Syncer(
            client=client,
            store=store,
            connections=args.connections,
            success_delay=args.success_delay,
        )
        await syncer.sync_all()


def main() -> None:
    with suppress(KeyboardInterrupt):
        asyncio.run(_main())


if __name__ == "__main__":
    main()
