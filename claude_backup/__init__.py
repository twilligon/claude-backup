"""Unofficial, unsanctioned tool to backup Claude.ai chats to local files."""

# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitOverride=false, reportUnusedCallResult=false

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
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
from typing import Any, ClassVar, TypeAlias, TypeVar, cast
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

__version__ = "0.1.6"

__all__ = ("__version__", "main", "Client", "Store", "Syncer")

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

    async def refresh(self, path: str) -> Json:
        retry_delay = self.min_retry_delay
        r = None
        for _ in range(self.retries):
            r = await self.session.get(
                f"https://claude.ai/api/{path}", allow_redirects=False
            )
            if r.ok:
                data = cast(Json, await r.json())
                return data
            else:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)

        assert r is not None  # never happens unless self.retries <= 0
        r.raise_for_status()
        raise ClientError  # pyright doesn't know above always throws. sad!


@dataclass(slots=True)
class Store:
    # migrate from unknown versions by not migrating anything, i.e. starting over
    MIGRATIONS: ClassVar[
        defaultdict[str | None, tuple[str, Callable[[str, str], None]]]
    ] = defaultdict(lambda: (__version__, lambda old, new: None))

    store_dir: Path
    force_refresh: bool = False

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

        (self.store_dir / "version").write_text(f"{__version__}\n")

    def save(self, path: Path, data: Json) -> None:
        cache_file = self.store_dir / path.with_suffix(".json")
        cache_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

        with NamedTemporaryFile(
            "w",
            prefix=f"{cache_file.name}-",
            dir=cache_file.parent,
        ) as f:
            json.dump(data, f)
            f.flush()
            Path(f.name).rename(cache_file)

    def load(self, path: Path) -> Json | None:
        if self.force_refresh:
            return None

        cache_file = self.store_dir / path.with_suffix(".json")
        with suppress(FileNotFoundError, NotADirectoryError), cache_file.open() as f:
            return cast(Json, json.load(f))

    def rename(self, old_path: Path, new_path: Path) -> None:
        old_file = self.store_dir / old_path.with_suffix(".json")
        new_file = self.store_dir / new_path.with_suffix(".json")
        new_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        old_file.rename(new_file)

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
    _store: Store    # pyright: ignore[reportUninitializedInstanceVariable]
    _data: Json      # pyright: ignore[reportUninitializedInstanceVariable]

    @property
    def client(self) -> Client:
        return self._client

    @property
    def store(self) -> Store:
        return self._store

    def set_data(self: T_APIObject, data: Json) -> T_APIObject:
        self._data = data
        return self

    def get_data(self) -> Json:
        return self._data

    @classmethod
    async def load(
        cls, *args: Any, api_path: str | None = None, store_path: Path | None = None
    ):
        obj = cls(*args)
        api_path = api_path or obj.api_path()
        store_path = store_path or obj.store_path()
        obj.set_data(obj.store.load(store_path) or await obj.client.refresh(api_path))
        return obj

    def api_path(self) -> str:
        raise NotImplementedError

    def store_path(self) -> Path:
        raise NotImplementedError

    async def refresh(self: T_APIObject) -> T_APIObject:
        return self.set_data(await self.client.refresh(self.api_path()))

    def save(self) -> None:
        self.store.save(self.store_path(), self.get_data())

    def __hash__(self) -> int:
        return hash(json.dumps(self._data, sort_keys=True, check_circular=False))

    def __eq__(self, other: object) -> bool:
        if type(self) is type(other):
            return self._data == cast("APIObject", other)._data
        else:
            return NotImplemented


class Nameable(APIObject):
    __slots__: ClassVar[tuple[()]] = ()

    _data: JsonD  # pyright: ignore[reportIncompatibleVariableOverride]

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


class Chat(Nameable):
    __slots__: ClassVar[tuple[str, str]] = (  # pyright: ignore[reportIncompatibleVariableOverride]
        "chat_conversations",
        "_data",
    )

    chat_conversations: "ChatConversationsList"
    _data: JsonD

    def __init__(
        self,
        chat_conversations: "ChatConversationsList",
    ):
        self.chat_conversations = chat_conversations

    @property
    def client(self) -> Client:
        return self.chat_conversations.client

    @property
    def store(self) -> Store:
        return self.chat_conversations.store

    def api_path(self) -> str:
        return (
            f"{self.chat_conversations.api_path()}/{self.uuid}"
            + "?tree=True&rendering_mode=messages&render_all_tools=true"
        )

    def store_path(self) -> Path:
        return self.chat_conversations.store_path() / self.slug()

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.chat_conversations))

    def __eq__(self, other: object) -> bool:
        return (
            super().__eq__(other)
            and self.chat_conversations == cast(Chat, other).chat_conversations
        )


class ChatConversationsEntry(Nameable):
    __slots__: ClassVar[tuple[str, str]] = (  # pyright: ignore[reportIncompatibleVariableOverride]
        "chat_conversations",
        "_data",
    )

    chat_conversations: "ChatConversationsList"
    _data: JsonD

    def __init__(
        self,
        chat_conversations: "ChatConversationsList",
    ):
        self.chat_conversations = chat_conversations

    @property
    def client(self) -> Client:
        return self.chat_conversations.client

    @property
    def store(self) -> Store:
        return self.chat_conversations.store

    def chat_api_path(self) -> str:
        return (
            f"{self.chat_conversations.api_path()}/{self.uuid}"
            + "?tree=True&rendering_mode=messages&render_all_tools=true"
        )

    def chat_store_path(self) -> Path:
        return self.chat_conversations.store_path() / self.slug()

    async def chat(self) -> Chat:
        return await Chat.load(
            self.chat_conversations,
            api_path=self.chat_api_path(),
            store_path=self.chat_store_path(),
        )

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.chat_conversations))

    def __eq__(self, other: object) -> bool:
        return (
            super().__eq__(other)
            and self.chat_conversations
            == cast(ChatConversationsEntry, other).chat_conversations
        )


class ChatConversationsList(APIObject):
    __slots__: ClassVar[tuple[str, str, str]] = ("organization", "unseen", "_data")

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

    def set_data(self, data: Json) -> "ChatConversationsList":
        # convert list (reverse chronological from API) to dict (forward chronological)
        entries_list = cast(list[JsonD], data)
        self._data = {  # pyright: ignore[reportIncompatibleVariableOverride]
            cast(str, entry["uuid"]): entry for entry in reversed(entries_list)
        }
        return self

    def get_data(self) -> Json:
        # convert dict (forward chronological) back to list (reverse chronological)
        return list(reversed(self._data.values()))

    def entry(self, uuid: str) -> ChatConversationsEntry | None:
        if data := self._data.get(uuid):
            return ChatConversationsEntry(self).set_data(data)
        return None

    def cached_entries(self) -> Iterator[ChatConversationsEntry]:
        # yield in reverse chronological order (newest first)
        for chat_data in reversed(self._data.values()):
            yield ChatConversationsEntry(self).set_data(chat_data)

    async def entries(self) -> AsyncGenerator[ChatConversationsEntry, None]:
        seen: set[ChatConversationsEntry] = set()

        async for entry in self.new_entries():
            seen.add(entry)
            yield entry

        for entry in self.cached_entries():
            if entry not in seen:
                yield entry

    async def new_entries(
        self, page_size: int = 20
    ) -> AsyncGenerator[ChatConversationsEntry, None]:
        # "sliding window sync" (chat_conversations is recently-modified-first)
        new: dict[str, JsonD] = {}

        offset = 0
        limit = self.unseen + 1 if self.unseen else page_size
        self.unseen = 0

        if not limit:
            return

        while True:
            page = cast(
                list[JsonD],
                await self.client.refresh(
                    f"{self.api_path()}?limit={limit}&offset={offset}"
                ),
            )

            done = False
            for chat in page:
                uuid = cast(str, chat["uuid"])

                if new_chat := new.get(uuid):
                    # this api doesn't have cursors or snapshots or anything :(
                    # so if a chat is created or updated *between our fetching
                    # one page and the next*, there is a *new* most recent chat
                    # which bumps all the others down and causes the last chat
                    # of the prior page also to be the first chat of the next:
                    #
                    # page 1 sees [A B]: [A B] C D E
                    # chat D is updated: D A B C E
                    # page 2 sees [B C]: D A [B C] E
                    #
                    # of course this can happen multiple times, and in fact the
                    # number of times it happens is exactly how many new chats
                    # we expect would be at offset 0 were we to fetch again. so
                    # update self.unseen as a hint to future new_entries calls
                    # that there are likely exactly self.unseen new or changed
                    self.unseen += 1

                    # ...but if the above hypothesis is wrong for some perverse
                    # reason like all the chats up until now being rewritten in
                    # a specific order behind our backs, still yield the chat:
                    #
                    # page 1 sees [A B]:  [A B] C D E
                    # B, D, & A updated:  A' D' B' C E
                    # page 2 sees [B' C]: A' D' [B' C] E
                    #
                    # so in case the cartesian daemon of claude chats is out to
                    # get us, we need to yield B' even if we wouldn't yield B
                    if chat == new_chat:
                        continue
                elif chat == self._data.get(uuid):
                    # chat was in a chronological list of chats we already had,
                    # so we must also have all chats before it, so we are done
                    done = True
                    break

                new[uuid] = chat
                yield ChatConversationsEntry(self).set_data(chat)

            # we *ought* to break from this loop by seeing something from prior
            # refreshes, but just in case e.g. all chats in self._data are gone
            # on claude.ai (or more likely moved/counted in self.unseen) we're
            # also definitely done if they ran out of items for this page
            if done or len(page) < limit:
                break

            # double it and give it to the next person
            offset += limit
            limit *= 2

        # for efficient inserts, self._data is in "forward-chronological" order
        # which means we need to merge reverse-chronological new in reverse. ez
        self._data.update(reversed(new.items()))
        self.save()

    async def refresh(self) -> "ChatConversationsList":
        async for _ in self.new_entries():
            pass
        return self

    def __len__(self) -> int:
        return len(self._data)

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.organization))

    def __eq__(self, other: object) -> bool:
        return (
            super().__eq__(other)
            and self.organization == cast(ChatConversationsList, other).organization
        )


class Organization(Nameable):
    __slots__: ClassVar[tuple[str, str, str]] = (  # pyright: ignore[reportIncompatibleVariableOverride]
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

    async def chat_conversations(self) -> ChatConversationsList:
        return await ChatConversationsList.load(self)


class Membership(APIObject):
    __slots__: ClassVar[tuple[str, str]] = (
        "account",
        "_data",  # pyright: ignore[reportIncompatibleVariableOverride]
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

    @property
    def organization(self) -> Organization:
        return Organization(self.client, self.store).set_data(
            cast(JsonD, self._data["organization"])
        )

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.account))

    def __eq__(self, other: object) -> bool:
        return super().__eq__(other) and self.account == cast(Membership, other).account


class Account(Nameable):
    __slots__: ClassVar[tuple[str, str, str]] = (  # pyright: ignore[reportIncompatibleVariableOverride]
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
        for m_data in cast(list[JsonD], self._data["memberships"]):
            yield Membership(self).set_data(m_data)


def truncate(s: str, max_len: int) -> str:
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


async def aroundrobin(*iterators: AsyncGenerator[T, None]) -> AsyncGenerator[T, None]:
    """Async round-robin across multiple async iterators."""
    try:
        done = False
        while not done:
            done = True
            for it in iterators:
                with suppress(StopAsyncIteration):
                    yield await anext(it)
                    done = False
    finally:
        await asyncio.gather(*(it.aclose() for it in iterators), return_exceptions=True)


@dataclass(slots=True)
class Syncer:
    client: Client
    store: Store
    connections: int = 6
    success_delay: float = 0.1
    tty: bool = field(default_factory=sys.stdout.isatty)

    async def _as_completed(
        self, awaitables: AsyncIterator[Awaitable[T]]
    ) -> AsyncGenerator[asyncio.Task[T], None]:
        pending: set[asyncio.Task[T]] = set()

        for _ in range(self.connections):
            try:
                awaitable = await anext(awaitables)
                pending.add(asyncio.ensure_future(awaitable))
            except StopAsyncIteration:
                break

        try:
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

            await asyncio.gather(*pending, return_exceptions=True)

            raise

    def as_completed(
        self, awaitables: Iterable[Awaitable[T]] | AsyncIterator[Awaitable[T]]
    ) -> AsyncGenerator[asyncio.Task[T], None]:
        async def asyncify(
            iterable: Iterable[Awaitable[T]],
        ) -> AsyncGenerator[Awaitable[T], None]:
            for item in iterable:
                yield item

        if hasattr(awaitables, "__anext__"):
            return self._as_completed(cast(AsyncIterator[Awaitable[T]], awaitables))
        else:
            return self._as_completed(
                asyncify(cast(Iterable[Awaitable[T]], awaitables))
            )

    async def gather(self, *awaitables: Awaitable[T]) -> list[T]:
        tasks_list = [asyncio.ensure_future(a) for a in awaitables]

        async with aclosing(self.as_completed(tasks_list)) as gen:
            async for task in gen:
                await task

        return [task.result() for task in tasks_list]

    async def get_chat_conversations_lists(self) -> list[ChatConversationsList]:
        account = await Account.load(self.client, self.store)
        account.save()

        async def get_chat_conversations_list(
            organization: Organization,
        ) -> ChatConversationsList:
            return await organization.chat_conversations()

        chat_conversations_list_fetches: list[Awaitable[ChatConversationsList]] = []
        for membership in account.memberships():
            organization = membership.organization
            if "chat" not in organization.capabilities:
                print(f'Skipping organization {organization} without "chat" capability')
                continue

            print(f"Fetching chats for organization {organization}")
            chat_conversations_list_fetches.append(
                get_chat_conversations_list(organization)
            )

        return await self.gather(*chat_conversations_list_fetches)

    def print_chat_entry(self, entry: ChatConversationsEntry) -> None:
        name = entry.name or ""
        if self.tty:
            try:
                width = os.get_terminal_size().columns
            except OSError:
                width = 80
            name = truncate(name, width - 36 - 4)
        print(f"{entry.uuid}\t{name}")

    async def get_chats(self) -> AsyncIterator[Awaitable[Chat]]:
        chat_conversations_lists = await self.get_chat_conversations_lists()

        async with aclosing(
            aroundrobin(*(ccl.entries() for ccl in chat_conversations_lists))
        ) as entries:
            async for entry in entries:
                self.print_chat_entry(entry)
                yield entry.chat()

        while needs_refresh := [ccl for ccl in chat_conversations_lists if ccl.unseen]:
            async with aclosing(
                aroundrobin(*(ccl.new_entries(page_size=1) for ccl in needs_refresh))
            ) as entries:
                async for entry in entries:
                    self.print_chat_entry(entry)
                    yield entry.chat()

    async def sync_all(self) -> None:
        async with aclosing(self.as_completed(self.get_chats())) as tasks:
            async for task in tasks:
                chat = await task
                old_entry = chat.chat_conversations.entry(chat.uuid)
                chat.save()
                if old_entry and old_entry.slug() != chat.slug():
                    chat.store.delete(old_entry.chat_store_path())


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
        "-s",
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
