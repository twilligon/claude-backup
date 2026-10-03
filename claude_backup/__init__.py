"""Unofficial, unsanctioned tool to back up Claude.ai chats to local files."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from asyncio import Task
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Generator,
    Iterable,
    Iterator,
    Sequence,
)
from contextlib import aclosing, contextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO, TextIOWrapper
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import TracebackType
from typing import IO, Any, ClassVar, TypeAlias, TypeVar, cast, final
from urllib.parse import quote
from uuid import UUID
import asyncio
import json
import os
import shutil
import sys

from fake_useragent import UserAgent
from platformdirs import user_data_dir
from aiohttp import (
    ClientResponseError,
    ClientSession,
    ClientTimeout,
    CookieJar,
    TCPConnector,
)
from yarl import URL

__version__ = "0.1.16"

__all__ = (
    "__version__",
    "main",
    "run",
    "Client",
    "Store",
    "Syncer",
    "APIObject",
    "Immutable",
    "Nameable",
    "Timestamped",
    "Account",
    "Membership",
    "Organization",
    "Chats",
    "ChatsEntry",
    "Chat",
    "File",
    "Files",
    "Json",
)


Json: TypeAlias = dict[str, "Json"] | list["Json"] | str | int | float | bool | None


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

    async def _download(self, path: str, sink: IO[bytes]) -> None:
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
            if not 200 <= r.status < 300:
                raise ClientResponseError(
                    r.request_info,
                    r.history,
                    status=r.status,
                    message=await r.text() if r.status == 404 else r.reason or "",
                    headers=r.headers,
                )

            if r.content_length is not None:
                sink.truncate(r.content_length)
            async for chunk in r.content.iter_any():
                sink.write(chunk)

    async def download(self, path: str, sink: IO[bytes]) -> None:
        retry_delay = self.min_retry_delay
        for retry in range(self.retries - 1):
            try:
                await self._download(path, sink)
                return
            except Exception as e:
                if isinstance(e, ClientResponseError) and e.status == 404:
                    raise

                sink.seek(0)
                sink.truncate()

                print(
                    f"Error fetching {path} (try {retry+1} of {self.retries}, "
                    + f"waiting {retry_delay:g}s): {e}",
                    file=sys.stderr,
                )

            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, self.max_retry_delay)

        await self._download(path, sink)

    async def refresh(self, path: str) -> dict[str, Json]:
        with BytesIO() as sink:
            await self.download(path, sink)
            return cast(dict[str, Json], json.loads(sink.getvalue()))


JSON_ARGS: dict[str, Any] = {
    "ensure_ascii": False,
    "check_circular": False,
    "separators": (",", ":"),
}


T_APIObject = TypeVar("T_APIObject", bound="APIObject")


@dataclass(slots=True)
class Store:
    COMPAT_VERSION: ClassVar[str] = "0.1.13"

    backup_dir: Path
    store_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.store_dir = self.backup_dir / self.COMPAT_VERSION
        self.backup_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.store_dir.mkdir(mode=0o700, exist_ok=True)

    def rename(self, old_path: Path, new_path: Path) -> bool:
        old_dir = self.store_dir / old_path
        new_dir = self.store_dir / new_path
        if not old_dir.exists() or new_dir.exists():
            return False

        new_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        old_dir.rename(new_dir)
        return True

    @contextmanager
    def save(
        self, path: Path, mtime: datetime | None = None
    ) -> Generator[IO[bytes], None, None]:
        cache_file = self.store_dir / path
        cache_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

        f = NamedTemporaryFile(
            "wb",
            prefix=f"{cache_file.name}-",
            dir=cache_file.parent,
            delete=False,
        )
        try:
            with f:
                yield f
            if cache_file.is_dir():
                shutil.rmtree(cache_file)
            Path(f.name).replace(cache_file)
        except BaseException:
            Path(f.name).unlink(missing_ok=True)
            raise

        if mtime is not None:
            os.utime(cache_file, (mtime.timestamp(), mtime.timestamp()))

    def load(self, path: Path) -> dict[str, Json] | None:
        cache_file = self.store_dir / path
        with (
            suppress(FileNotFoundError, NotADirectoryError),
            cache_file.open(encoding="utf-8") as f,
        ):
            return cast(dict[str, Json], json.load(f))

    def mtime(self, path: Path) -> datetime | None:
        with suppress(FileNotFoundError, NotADirectoryError):
            return datetime.fromtimestamp(
                (self.store_dir / path).stat().st_mtime, timezone.utc
            )
        return None

    def delete(self, path: Path) -> None:
        file = self.store_dir / path
        with suppress(FileNotFoundError):
            file.unlink()

        parent = file.parent
        while parent != self.store_dir:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


class APIObject:
    __slots__: tuple[str, ...] = ("_data",)

    _client: Client  # pyright: ignore[reportUninitializedInstanceVariable]
    _store: Store  # pyright: ignore[reportUninitializedInstanceVariable]
    _data: dict[str, Json]  # pyright: ignore[reportUninitializedInstanceVariable]

    @property
    def client(self) -> Client:
        return self._client

    @property
    def store(self) -> Store:
        return self._store

    def get_data(self) -> dict[str, Json]:
        return self._data

    def get_mtime(self) -> datetime | None:
        return None

    def set_data(self: T_APIObject, data: dict[str, Json]) -> T_APIObject:
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
        path = store_path or obj.store_path()
        data = obj.store.load(path.with_name(path.name + ".json"))
        if data is not None:
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

    def save(self: T_APIObject) -> T_APIObject:
        path = self.store_path()
        with self.store.save(
            path.with_name(path.name + ".json"), self.get_mtime()
        ) as f:
            with TextIOWrapper(f, encoding="utf-8") as text:
                json.dump(self.get_data(), text, **JSON_ARGS)
        return self

    def delete_cached(self) -> None:
        path = self.store_path()
        self.store.delete(path.with_name(path.name + ".json"))


class Immutable(APIObject):
    __slots__: tuple[str, ...] = ()

    def __hash__(self) -> int:
        return hash(json.dumps(self._data, sort_keys=True, **JSON_ARGS))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self._data == other._data
        return NotImplemented


class Nameable(APIObject):
    __slots__: tuple[str, ...] = ()

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

    def print(self) -> None:
        name = self.name or ""

        if sys.stdout.isatty():
            max_len = shutil.get_terminal_size().columns - 36 - 4
            if len(name) > max_len:
                name = name[: max_len - 1] + "…"

        print(f"{self.uuid}\t{name}")


class Timestamped(APIObject):
    __slots__: tuple[str, ...] = ()

    @property
    def created_at(self) -> datetime:
        # the .replace("Z", ...) is for Python <3.11, which doesn't accept
        # the trailing-Z form fromisoformat() that claude.ai uses
        return datetime.fromisoformat(
            cast(str, self._data["created_at"]).replace("Z", "+00:00")
        )

    @property
    def updated_at(self) -> datetime:
        return datetime.fromisoformat(
            cast(str, self._data["updated_at"]).replace("Z", "+00:00")
        )

    def get_mtime(self) -> datetime:
        return self.updated_at


@final
class Chat(Timestamped, Nameable):
    __slots__ = ("chat_list",)

    chat_list: "Chats"

    def __init__(self, chat_list: "Chats"):
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
            + "&return_dangling_human_message=true&include_inline_comparison=true"
            + "&consistency=strong"
        )

    def store_path(self) -> Path:
        return self.chat_list.store_path() / self.slug()


@final
class File(Timestamped, Nameable):
    __slots__ = ("file_list",)

    file_list: "Files"

    def __init__(self, file_list: "Files"):
        self.file_list = file_list

    @property
    def client(self) -> Client:
        return self.file_list.client

    @property
    def store(self) -> Store:
        return self.file_list.store

    @property
    def uuid(self) -> str:
        return str(UUID(cast(str, self._data["file_uuid"])))

    @property
    def name(self) -> str | None:
        return cast(str, self._data.get("file_name")) or None

    def get_mtime(self) -> datetime:
        return self.created_at

    def api_path(self) -> str:
        return f"{self.file_list.organization.api_path()}/files/{self.uuid}/contents"

    def store_path(self) -> Path:
        return self.file_list.store_path() / self.slug()

    async def fetch(self) -> "File":
        with self.store.save(self.store_path(), self.get_mtime()) as f:
            try:
                await self.client.download(self.api_path(), f)
            except ClientResponseError as e:
                if e.status == 404:
                    # TODO: slightly jank
                    await self.client.download(
                        f"{self.file_list.organization.uuid}/files/{self.uuid}/preview",
                        f,
                    )
                else:
                    raise
        return self


@final
class Files(APIObject):
    __slots__ = ("organization",)

    organization: "Organization"

    def __init__(self, organization: "Organization"):
        self.organization = organization
        self._data = {}

    @property
    def client(self) -> Client:
        return self.organization.client

    @property
    def store(self) -> Store:
        return self.organization.store

    def api_path(self) -> str:
        return f"{self.organization.api_path()}/account/files"

    def store_path(self) -> Path:
        return self.organization.store_path() / "files"

    async def entries(self) -> AsyncGenerator[File, None]:
        new: dict[str, File] = {}
        query = "?limit=100"

        while True:
            response = await self.client.refresh(self.api_path() + query)

            done = False
            for raw in cast(list[dict[str, Json]], response["files"]):
                file = File(self).set_data(raw)
                if file.uuid in self._data:
                    done = True
                    break

                new[file.uuid] = file
                yield file

            if done or response["next_cursor"] is None:
                break

            cursor = quote(cast(str, response["next_cursor"]), safe="")
            query = f"?limit=100&cursor={cursor}"

        for uuid, file in reversed(new.items()):
            self._data[uuid] = file._data
        self.save()

        for uuid, raw in reversed(self._data.items()):
            if uuid not in new:
                yield File(self).set_data(cast(dict[str, Json], raw))


@final
class ChatsEntry(Timestamped, Nameable, Immutable):
    __slots__ = ("chat_list",)

    chat_list: "Chats"

    def __init__(self, chat_list: "Chats"):
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
            + "&return_dangling_human_message=true&include_inline_comparison=true"
            + "&consistency=strong"
        )

    def chat_store_path(self) -> Path:
        return self.chat_list.store_path() / self.slug()

    def load_chat(self) -> Chat | None:
        return Chat._load(self.chat_list, store_path=self.chat_store_path())

    def chat_mtime(self) -> datetime | None:
        path = self.chat_store_path()
        return self.store.mtime(path.with_name(path.name + ".json"))

    async def fetch_chat(self) -> Chat:
        return await Chat._fetch(self.chat_list, api_path=self.chat_api_path())


@final
class Chats(APIObject):
    __slots__ = ("organization", "unseen")

    organization: "Organization"
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
        return self.organization.store_path() / "chats"

    def entry(self, uuid: str) -> ChatsEntry | None:
        if (raw := self._data.get(uuid)) is not None:
            return ChatsEntry(self).set_data(cast(dict[str, Json], raw))
        return None

    def cached_entries(self) -> Iterator[ChatsEntry]:
        # yield in reverse chronological order (newest first)
        for raw in reversed(self._data.values()):
            yield ChatsEntry(self).set_data(cast(dict[str, Json], raw))

    async def new_entries(
        self, page_size: int = 30
    ) -> AsyncGenerator[ChatsEntry, None]:
        # "sliding window" sync (chat_conversations is recently-modified-first)
        new: dict[str, ChatsEntry] = {}

        if not self._data:
            # first fetch: grab everything in one unpaginated request (yes, the
            # api really does work that way, insanity)
            response = await self.client.refresh(
                f"{self.organization.api_path()}/chat_conversations_v2?consistency=strong"
            )
            for raw in reversed(cast(list[dict[str, Json]], response["data"])):
                self._data[ChatsEntry(self).set_data(raw).uuid] = raw
            for entry in self.cached_entries():
                yield entry
            self.save()
            old_path = self.organization.store_path()
            self.store.delete(old_path.with_name(old_path.name + ".json"))
            return

        offset = 0
        limit = self.unseen + 1 if self.unseen else page_size
        self.unseen = 0
        last: dict[str, Json] | None = None

        assert limit > 0

        while True:
            response = await self.client.refresh(
                f"{self.organization.api_path()}/chat_conversations_v2"
                + f"?limit={limit}&offset={offset}&consistency=strong"
            )
            page = cast(list[dict[str, Json]], response["data"])
            has_more = cast(bool, response["has_more"])

            skip = 0
            if last is not None:
                for skip, raw in enumerate(page):
                    if raw["uuid"] == last["uuid"]:
                        if raw["updated_at"] != last["updated_at"]:
                            self.unseen = offset + skip
                        break
                    else:
                        self.unseen += 1
                else:
                    offset = 0
                    last = None
                    self.unseen = 0
                    continue

            done = False
            for raw in page[skip:]:
                entry = ChatsEntry(self).set_data(raw)
                uuid = entry.uuid

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
                    del new[uuid]
                    if entry.updated_at == new_entry.updated_at:
                        new[uuid] = entry
                        continue
                elif (
                    stored := self._data.get(uuid)
                ) and entry.updated_at == ChatsEntry(self).set_data(
                    cast(dict[str, Json], stored)
                ).updated_at:
                    # entry was in a chronological list of ones we already had,
                    # so we must also have all entries before it, so we're done
                    self._data[uuid] = raw
                    done = True
                    continue

                new[uuid] = entry
                yield entry

            # we *ought* to break from this loop by seeing something from prior
            # refreshes, but just in case e.g. all seen entries were deleted on
            # claude.ai (or more likely moved/counted in self.unseen)... we are
            # also definitely done if they say there are no more items
            if done or not has_more:
                break

            # if not, double it and give it to the next person
            if page:
                offset += len(page) - 1
                last = page[-1]
            limit *= 2

        # dicts preserve order, but because there is no dict.prepend(), we need
        # self._data to be in "forward-chronological" order so newly discovered
        # chats belong at the "end" w/r/t iteration order. thus we insert items
        # from necessarily reverse-chronological new in reverse so self._data's
        # still entirely in chronological order. this may be a bit galaxy brain
        for uuid, entry in reversed(new.items()):
            self._data.pop(uuid, None)
            self._data[uuid] = entry._data
        self.save()

    async def entries(self) -> AsyncGenerator[ChatsEntry, None]:
        seen: set[ChatsEntry] = set()
        while True:
            async for entry in self.new_entries():
                seen.add(entry)
                yield entry
            if not self.unseen:
                break
        for entry in self.cached_entries():
            if entry not in seen:
                yield entry


@final
class Organization(Nameable):
    __slots__ = ("account",)

    account: "Account"

    def __init__(self, account: "Account"):
        self.account = account

    @property
    def client(self) -> Client:
        return self.account.client

    @property
    def store(self) -> Store:
        return self.account.store

    def api_path(self) -> str:
        return f"organizations/{self.uuid}"

    def store_path(self) -> Path:
        return Path(self.account.slug()) / self.slug()

    @property
    def capabilities(self) -> list[str]:
        return cast(list[str], self._data["capabilities"])

    def chat_list(self) -> Chats:
        return Chats._load(self) or Chats(self)

    def file_list(self) -> Files:
        return Files._load(self) or Files(self)


@final
class Membership(Immutable):
    __slots__ = ("account",)

    account: "Account"

    def __init__(self, account: "Account"):
        self.account = account

    @property
    def client(self) -> Client:
        return self.account.client

    @property
    def store(self) -> Store:
        return self.account.store

    def organization(self) -> Organization:
        return Organization(self.account).set_data(
            cast(dict[str, Json], self._data["organization"])
        )


@final
class Account(Nameable):
    __slots__ = ("_client", "_store")

    _client: Client
    _store: Store

    def __init__(self, client: Client, store: Store):
        self._client = client
        self._store = store

    @property
    def name(self) -> str | None:
        return cast(str, self._data.get("email_address")) or None

    def api_path(self) -> str:
        return "account"

    def store_path(self) -> Path:
        return Path(self.slug())

    @classmethod
    def load(cls, client: Client, store: Store) -> Iterator["Account"]:
        for cache_file in store.store_dir.glob("*-*.json"):
            store_path = Path(cache_file.name.removesuffix(".json"))
            if account := cls._load(client, store, store_path=store_path):
                yield account

    def memberships(self) -> Iterator[Membership]:
        for membership in cast(list[dict[str, Json]], self._data["memberships"]):
            yield Membership(self).set_data(membership)

    def organization(self, uuid: str) -> Organization | None:
        for membership in self.memberships():
            organization = membership.organization()
            if organization.uuid == uuid:
                return organization

        return None


T = TypeVar("T")


async def as_completed(
    source: AsyncGenerator[Awaitable[T], None],
    limit: int,
    delay: float = 0.0,
) -> AsyncGenerator[Task[T], None]:
    pending: set[Task[T]] = set()

    try:
        for _ in range(limit):
            try:
                awaitable = await anext(source)
                pending.add(asyncio.ensure_future(awaitable))
            except StopAsyncIteration:
                break

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                yield task

                await asyncio.sleep(delay)

                try:
                    awaitable = await anext(source)
                    pending.add(asyncio.ensure_future(awaitable))
                except StopAsyncIteration:
                    pass
    except BaseException:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        raise
    finally:
        await source.aclose()


@dataclass(slots=True)
class Syncer:
    client: Client
    store: Store
    connections: int = 1
    success_delay: float = 0.25

    async def get_organizations(self) -> AsyncGenerator[Organization]:
        account = Account(self.client, self.store)
        account.set_data(await self.client.refresh(account.api_path()))

        old_account = next(
            (
                old
                for old in Account.load(self.client, self.store)
                if old.uuid == account.uuid
            ),
            None,
        )
        if old_account and old_account.slug() != account.slug():
            print(
                f"Renaming account {old_account} to {account.name or account.uuid}",
                file=sys.stderr,
            )
            new_path = account.store_path()
            old_path = new_path.with_name(old_account.slug())
            self.store.rename(old_path, new_path)
            self.store.rename(
                old_path.with_name(old_path.name + ".json"),
                new_path.with_name(new_path.name + ".json"),
            )
        account.save()

        for membership in account.memberships():
            organization = membership.organization()

            if (
                old_account
                and (old_organization := old_account.organization(organization.uuid))
                and old_organization.slug() != organization.slug()
            ):
                print(
                    f"Renaming organization {old_organization} to {organization.name or organization.uuid}",
                    file=sys.stderr,
                )
                new_path = organization.store_path()
                old_path = new_path.with_name(old_organization.slug())
                self.store.rename(old_path, new_path)
                self.store.delete(old_path.with_name(old_path.name + ".json"))

            if "chat" not in organization.capabilities:
                print(
                    f'Skipping organization {organization} without "chat" capability',
                    file=sys.stderr,
                )
                continue

            print(f"Fetching chats for organization {organization}", file=sys.stderr)
            yield organization

    async def fetches(
        self,
    ) -> AsyncGenerator[Awaitable[Chat | ChatsEntry | File], None]:
        async def fetch_new_chat(
            entry: ChatsEntry, old_chat: Chat | None
        ) -> Chat | ChatsEntry:
            try:
                chat = await entry.fetch_chat()
            except ClientResponseError as e:
                if e.status == 404 and "chat_conversation_not_found" in e.message:
                    return entry
                else:
                    raise

            if old_chat and old_chat.store_path() != chat.store_path():
                old_chat.delete_cached()

            return chat

        async for organization in self.get_organizations():
            async for entry in organization.chat_list().entries():
                # we must get old_entry *now* and not in the async function in
                # fetch_new_chat, since Chats.new_entries might finish and save
                # the new entry over old_entry!
                old_entry = entry.chat_list.entry(entry.uuid)
                if old_entry and old_entry.chat_mtime() == entry.updated_at:
                    continue
                old_chat = old_entry.load_chat() if old_entry else None
                if old_chat and old_chat.updated_at == entry.updated_at:
                    continue

                entry.print()
                yield fetch_new_chat(entry, old_chat)

            async for file in organization.file_list().entries():
                if not (self.store.store_dir / file.store_path()).is_file():
                    file.print()
                    yield file.fetch()

    async def sync_all(self) -> None:
        gone: list[ChatsEntry] = []

        async with aclosing(
            as_completed(self.fetches(), self.connections, self.success_delay)
        ) as tasks:
            async for task in tasks:
                result = await task
                if isinstance(result, ChatsEntry):
                    gone.append(result)

        for entry in gone:
            entry.chat_list.get_data().pop(entry.uuid, None)
        for chat_list in {entry.chat_list for entry in gone}:
            chat_list.save()


@dataclass(slots=True)
class DefaultPath:
    path: str

    def __str__(self):
        try:
            return f"~/{Path(self.path).relative_to(Path.home())}"
        except ValueError:
            return self.path


async def _run(argv: Sequence[str], keys: Iterable[str]) -> None:
    def default(cls: type[Any], key: str) -> Any:
        return cls.__dataclass_fields__[key].default

    parser = ArgumentParser(
        prog="claude-backup",
        description="Back up Claude.ai chats",
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
    args = parser.parse_args(argv)
    if isinstance(args.backup_dir, DefaultPath):
        args.backup_dir = args.backup_dir.path

    store = Store(Path(args.backup_dir))

    for key in keys:
        async with Client(
            session_key=key,
            retries=args.retries,
            min_retry_delay=args.min_retry_delay,
            max_retry_delay=args.max_retry_delay,
        ) as client:
            await Syncer(
                client=client,
                store=store,
                connections=args.connections,
                success_delay=args.success_delay,
            ).sync_all()


def run(argv: Sequence[str], keys: Iterable[str]) -> None:
    asyncio.run(_run(argv, keys))


def main() -> None:
    def keys() -> Iterator[str]:
        for key in filter(None, map(str.strip, sys.stdin)):
            if not key.startswith("sk-"):
                sys.exit(f"not a sessionKey: {key}")
            yield key

    run(sys.argv[1:], keys())
