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
from contextlib import aclosing, suppress
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from types import TracebackType
from typing import Any, ClassVar, TypeAlias, TypeVar, cast
from uuid import UUID
import asyncio
import hashlib
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

__all__ = ("__version__", "main", "Client", "Store", "Syncer")

Json: TypeAlias = dict[str, "Json"] | list["Json"] | str | int | float | bool | None
JsonD: TypeAlias = dict[str, Json]

T = TypeVar("T")


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
        r.raise_for_status()  # pyright: ignore[reportOptionalMemberAccess]
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


class APIObject:
    __slots__ = ("client", "store", "_data")

    client: Client
    store: Store
    _data: Json

    def __init__(self, client: Client, store: Store, data: Json):
        self.client = client
        self.store = store
        self._data = data

    def api_path(self) -> str:
        raise NotImplementedError

    def store_path(self) -> Path:
        raise NotImplementedError

    def update(self, data: Json) -> None:
        self._data = data
        self.save()

    async def refresh(self) -> None:
        self.update(await self.client.refresh(self.api_path()))

    def save(self) -> None:
        self.store.save(self.store_path(), self._data)

    def load(self) -> bool:
        data = self.store.load(self.store_path())
        if data is not None:
            self._data = data
            return True
        return False

    def __hash__(self) -> int:
        return hash(json.dumps(self._data, sort_keys=True, check_circular=False))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, APIObject) and type(self) is type(other):
            return self._data == other._data
        return NotImplemented


class Nameable(APIObject):
    __slots__ = ()

    _data: JsonD  # pyright: ignore[reportIncompatibleVariableOverride]

    FILENAME_XLAT: ClassVar[dict[int, int]] = {
        ord(c): ord("_") for c in '<>:"|?*/\\ \t\n\r'
    }

    @property
    def uuid(self) -> str:
        return str(UUID(cast(str, self._data["uuid"])))

    @property
    def name(self) -> str | None:
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
    __slots__ = ("chat_conversations",)

    def __init__(
        self,
        client: Client,
        store: Store,
        chat_conversations: "ChatConversations",
        data: JsonD,
    ):
        super().__init__(client, store, data)
        self.chat_conversations = chat_conversations

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
            isinstance(other, Chat)
            and super().__eq__(other)
            and self.chat_conversations == other.chat_conversations
        )


class ChatConversationsEntry(Nameable):
    __slots__ = ("chat_conversations",)

    def __init__(
        self,
        client: Client,
        store: Store,
        chat_conversations: "ChatConversations",
        data: JsonD,
    ):
        super().__init__(client, store, data)
        self.chat_conversations = chat_conversations

    def store_path(self) -> Path:
        return self.chat_conversations.store_path() / self.slug()

    async def chat(self) -> Chat:
        data = self.store.load(self.store_path())
        if data is None:
            data = await self.client.refresh(
                f"{self.chat_conversations.api_path()}/{self.uuid}"
                + "?tree=True&rendering_mode=messages&render_all_tools=true"
            )
            chat = Chat(
                self.client, self.store, self.chat_conversations, cast(JsonD, data)
            )
            chat.save()
            return chat
        return Chat(self.client, self.store, self.chat_conversations, cast(JsonD, data))

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.chat_conversations))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ChatConversationsEntry)
            and super().__eq__(other)
            and self.chat_conversations == other.chat_conversations
        )


class ChatConversations(APIObject):
    __slots__ = ("organization", "_unseen")

    _data: dict[str, JsonD]  # pyright: ignore[reportIncompatibleVariableOverride]
    _unseen: int

    def __init__(
        self, client: Client, store: Store, organization: "Organization", data: Json
    ):
        # Convert list (reverse chronological from API) to dict (forward chronological)
        entries_list = cast(list[JsonD], data)
        entries_dict = {
            cast(str, entry["uuid"]): entry for entry in reversed(entries_list)
        }
        super().__init__(client, store, entries_dict)
        self.organization = organization
        self._unseen = 0

    def store_path(self) -> Path:
        return self.organization.store_path() / "chat_conversations"

    def save(self) -> None:
        # Convert dict (forward chronological) back to list (reverse chronological)
        data_as_list = list(reversed(list(self._data.values())))
        self.store.save(self.store_path(), data_as_list)

    def cached_entries(self) -> Iterator[ChatConversationsEntry]:
        # Yield in reverse chronological order (newest first)
        for chat in reversed(list(self._data.values())):
            yield ChatConversationsEntry(self.client, self.store, self, chat)

    async def entries(self) -> AsyncGenerator[ChatConversationsEntry, None]:
        seen = set()

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
        limit = self._unseen + 1 if self._unseen else page_size
        self._unseen = 0
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
                    # update self._unseen as a hint to future new_entries calls
                    # that there are likely exactly self._unseen new or changed
                    self._unseen += 1

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
                yield ChatConversationsEntry(self.client, self.store, self, chat)

            # we *ought* to break from this loop by seeing something from prior
            # refreshes, but just in case e.g. all chats in self._data are gone
            # on claude.ai (or more likely moved/counted in self._unseen) we're
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

    async def refresh(self) -> None:
        async for _ in self.new_entries():
            pass

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[ChatConversationsEntry]:
        # Iterate in reverse chronological order (API order)
        for chat_data in reversed(list(self._data.values())):
            yield ChatConversationsEntry(self.client, self.store, self, chat_data)

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.organization))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ChatConversations)
            and super().__eq__(other)
            and self.organization == other.organization
        )


class Organization(Nameable):
    __slots__ = ()

    def api_path(self) -> str:
        return f"organizations/{self.uuid}"

    def store_path(self) -> Path:
        return Path("organizations") / self.slug()

    @property
    def capabilities(self) -> list[str]:
        return cast(list[str], self._data["capabilities"])

    async def chat_conversations(self) -> ChatConversations:
        data = self.store.load(self.store_path() / "chat_conversations")
        if data is None:
            data = await self.client.refresh(f"{self.api_path()}/chat_conversations")
        return ChatConversations(self.client, self.store, self, data)


class Membership(APIObject):
    __slots__ = ("account",)

    def __init__(self, client: Client, store: Store, account: "Account", data: Json):
        super().__init__(client, store, data)
        self.account = account

    @property
    def organization(self) -> Organization:
        return Organization(
            self.client,
            self.store,
            cast(JsonD, cast(JsonD, self._data)["organization"]),
        )

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.account))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Membership)
            and super().__eq__(other)
            and self.account == other.account
        )


class Account(Nameable):
    __slots__ = ()

    def api_path(self) -> str:
        return "account"

    def store_path(self) -> Path:
        return Path("account")

    def memberships(self) -> Iterator[Membership]:
        for m in cast(list[JsonD], self._data["memberships"]):
            yield Membership(self.client, self.store, self, m)


def truncate(s: str, max_len: int) -> str:
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


async def aroundrobin(*iterators: AsyncIterator[T]) -> AsyncIterator[T]:
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

    async def as_completed(
        self, awaitables: Iterable[Awaitable[T]]
    ) -> AsyncGenerator[asyncio.Task[T], None]:
        awaitables_iter = iter(awaitables)
        pending: set[asyncio.Task[T]] = set(
            map(asyncio.ensure_future, islice(awaitables_iter, self.connections))
        )

        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    yield task

                    await asyncio.sleep(self.success_delay)

                    if next_task := next(awaitables_iter, None):
                        pending.add(asyncio.ensure_future(next_task))
        except BaseException:  # catch *everything* so we can cancel tasks
            for task in pending:
                task.cancel()

            await asyncio.gather(*pending, return_exceptions=True)

            raise

    async def gather(self, *awaitables: Awaitable[T]) -> list[T]:
        awaitables_list = list(awaitables)

        async with aclosing(self.as_completed(awaitables_list)) as gen:
            async for task in gen:
                await task

        # gather all, in original order; we awaited above so this is instant
        return await asyncio.gather(*awaitables_list)

    async def get_chat_lists(self) -> list[AsyncIterator[ChatConversationsEntry]]:
        account = Account(self.client, self.store, await self.client.refresh("account"))
        account.save()

        async def get_entries(organization):
            return (await organization.chat_conversations()).entries()

        entry_iterator_fetches = []
        for membership in account.memberships():
            organization = membership.organization
            if "chat" not in organization.capabilities:
                print(f'Skipping organization {organization} without "chat" capability')
                continue

            print(f"Fetching chats for organization {organization}")
            entry_iterator_fetches.append(get_entries(organization))

        return await self.gather(*entry_iterator_fetches)

    async def get_chats(self) -> AsyncIterator[Awaitable[Chat]]:
        chat_lists = await self.get_chat_lists()

        tty = sys.stdout.isatty()
        async with aclosing(aroundrobin(*chat_lists)) as entries:
            async for entry in entries:
                name = entry.name or ""

                if tty:
                    try:
                        width = os.get_terminal_size().columns
                    except OSError:
                        width = 80
                    name = truncate(name, width - 36 - 4)  # width - uuid - tab
                print(f"{entry.uuid}\t{name}")
                yield entry.chat()

    async def sync_all(self) -> None:
        async with aclosing(self.get_chats()) as chats:
            async with aclosing(self.as_completed(chats)) as tasks:
                async for task in tasks:
                    await task


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
    def default(cls: type, key: str) -> Any:
        return cls.__dataclass_fields__[key].default

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
        default=default(Syncer, "connections"),
        help="Maximum concurrent connections",
    )
    parser.add_argument(
        "-s",
        "--success-delay",
        type=float,
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
        default=default(Client, "min_retry_delay"),
        help="Minimum retry delay in seconds",
    )
    parser.add_argument(
        "--max-retry-delay",
        type=float,
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
    asyncio.run(_main())


if __name__ == "__main__":
    main()
