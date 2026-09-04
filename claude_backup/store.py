# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitOverride=false, reportUnusedCallResult=false
# pyright: reportPrivateUsage=false, reportIncompatibleVariableOverride=false

from collections import defaultdict
from collections.abc import AsyncGenerator, Callable, Generator, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO, Any, ClassVar, TypeVar, cast, final
from uuid import UUID
import json
import os
import shutil
import sys

from . import __version__
from .client import Client, Json, JsonD

__all__ = (
    "Store",
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
    "Asset",
)


JSON_ARGS: dict[str, Any] = {
    "ensure_ascii": False,
    "check_circular": False,
    "separators": (",", ":"),
}


T_APIObject = TypeVar("T_APIObject", bound="APIObject")


@dataclass(slots=True)
class Store:
    def _wipe(self) -> None:
        for entry in self.store_dir.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()

    def _set_chat_mtimes(self) -> None:
        self._fix_bad_slug_paths()

        for account in Account.load(None, self):  # pyright: ignore[reportArgumentType]
            for membership in account.memberships():
                if chats := membership.organization().chat_list():
                    for entry in chats.cached_entries():
                        if chat := entry.load_chat():
                            chat.save()

    def _fix_bad_slug_paths(self) -> None:
        for account in Account.load(None, self):  # pyright: ignore[reportArgumentType]
            for membership in account.memberships():
                if chats := membership.organization().chat_list():
                    for entry in chats.cached_entries():
                        old_path = entry.chat_store_path().with_suffix("")
                        if entry.chat_store_path() != old_path and (
                            chat := Chat._load(chats, store_path=old_path)
                        ):
                            chat.save()
                            self.delete(old_path)

    MIGRATIONS: ClassVar[
        defaultdict[str | None, tuple[str, Callable[["Store"], None]]]
    ] = defaultdict(
        # migrate from unknown versions by not migrating anything/starting over
        lambda: (__version__, Store._wipe),
        {
            "0.1.7": ("0.1.8", lambda _: None),
            "0.1.8": ("0.1.9", _set_chat_mtimes),
            "0.1.9": ("0.1.10", lambda _: None),
            "0.1.10": ("0.1.11", lambda _: None),
            "0.1.11": ("0.1.12", _fix_bad_slug_paths),
        },
    )

    store_dir: Path

    def __post_init__(self):
        if not self.store_dir.exists():
            self.store_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            (self.store_dir / "version").write_text(f"{__version__}\n")
            return

        try:
            version = (self.store_dir / "version").read_text().strip()
        except FileNotFoundError:
            (self.store_dir / "version").write_text(f"{__version__}\n")
            return

        while version != __version__:
            version, migrate_step = self.MIGRATIONS[version]
            migrate_step(self)
            (self.store_dir / "version").write_text(f"{version}\n")

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
        self, path: Path, mtime: datetime | float | None = None
    ) -> Generator[IO[bytes], None, None]:
        cache_file = self.store_dir / path
        cache_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

        with NamedTemporaryFile(
            "wb",
            prefix=f"{cache_file.name}-",
            dir=cache_file.parent,
            delete=False,
        ) as f:
            try:
                yield f
                f.close()
                Path(f.name).rename(cache_file)
            except BaseException:
                Path(f.name).unlink(missing_ok=True)
                raise

        if mtime is not None:
            mtime = mtime.timestamp() if isinstance(mtime, datetime) else mtime
            os.utime(cache_file, (mtime, mtime))

    def find(self, path: Path) -> Path | None:
        cache_file = self.store_dir / path
        return cache_file if cache_file.is_file() else None

    def load(self, path: Path) -> Json | None:
        cache_file = self.store_dir / path.with_name(path.name + ".json")
        with suppress(FileNotFoundError, NotADirectoryError), cache_file.open() as f:
            return cast(Json, json.load(f))

    def delete(self, path: Path) -> None:
        file = self.store_dir / path.with_name(path.name + ".json")
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
    __slots__ = ("__weakref__",)

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

    def get_mtime(self) -> datetime | float | None:
        return None

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
        data = obj.store.load(store_path or obj.store_path())
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

    async def refresh(self: T_APIObject) -> T_APIObject:
        return self.set_data(await self.client.refresh(self.api_path())).save()

    def save(self: T_APIObject) -> T_APIObject:
        path = self.store_path()
        with self.store.save(
            path.with_name(path.name + ".json"), self.get_mtime()
        ) as f:
            with TextIOWrapper(f, encoding="utf-8") as text:
                json.dump(self.get_data(), text, **JSON_ARGS)
        return self

    def delete_cached(self) -> None:
        self.store.delete(self.store_path())


class Immutable(APIObject):
    __slots__ = ()

    def __hash__(self) -> int:
        return hash(json.dumps(self._data, sort_keys=True, **JSON_ARGS))

    def __eq__(self, other: object) -> bool:
        if type(self) is type(other):
            assert isinstance(other, APIObject)  # appease pyright
            return self._data == other._data
        else:
            return NotImplemented


class Nameable(APIObject):
    __slots__ = ()

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


class Timestamped(APIObject):
    __slots__ = ()

    _data: JsonD

    @property
    def created_at(self) -> datetime | None:
        try:
            # the .replace("Z", ...) is for Python <3.11, which doesn't accept
            # the trailing-Z form fromisoformat() that claude.ai uses
            return datetime.fromisoformat(
                cast(str, self._data["created_at"]).replace("Z", "+00:00")
            )
        except (KeyError, TypeError, ValueError):
            return None

    @property
    def updated_at(self) -> datetime | None:
        try:
            return datetime.fromisoformat(
                cast(str, self._data["updated_at"]).replace("Z", "+00:00")
            )
        except (KeyError, TypeError, ValueError):
            return None

    def get_mtime(self) -> datetime | float | None:
        return self.updated_at


@final
class Chat(Timestamped, Nameable):
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
        return self.chat_list.store_path() / "chats" / self.slug()

    def files(self) -> Iterator["File"]:
        seen: set[str] = set()
        for message in cast(list[JsonD], self._data.get("chat_messages") or []):
            for data in cast(list[JsonD], message.get("files") or []):
                file = File(self).set_data(data)
                if file.uuid not in seen:
                    seen.add(file.uuid)
                    yield file


@final
class File(Timestamped, Nameable):
    __slots__ = (
        "chat",
        "_data",
    )

    chat: Chat
    _data: JsonD

    def __init__(self, chat: Chat):
        self.chat = chat

    @property
    def client(self) -> Client:
        return self.chat.client

    @property
    def store(self) -> Store:
        return self.chat.store

    @property
    def name(self) -> str | None:
        return cast(str, self._data.get("file_name")) or None

    def get_mtime(self) -> datetime | float | None:
        return self.updated_at or self.created_at

    def api_path(self) -> str:
        return f"{self.chat.chat_list.organization.uuid}/files/{self.uuid}"

    def store_path(self) -> Path:
        return self.chat.chat_list.organization.store_path() / "files" / self.slug()

    def assets(self) -> Iterator["Asset"]:
        for key, data in self._data.items():
            if key.endswith("_asset"):
                yield Asset(self).set_data(cast(JsonD, data))


@final
class Asset(APIObject):
    __slots__ = (
        "file",
        "_data",
    )

    file: File
    _data: JsonD

    def __init__(self, file: File):
        self.file = file

    @property
    def client(self) -> Client:
        return self.file.client

    @property
    def store(self) -> Store:
        return self.file.store

    @property
    def variant(self) -> str:
        return cast(str, self._data["url"]).rsplit("/", 1)[-1]

    def __str__(self) -> str:
        return f"{self.variant} of {self.file}"

    def get_mtime(self) -> datetime | float | None:
        return self.file.get_mtime()

    def api_path(self) -> str:
        return f"{self.file.api_path()}/{self.variant}"

    def store_path(self) -> Path:
        return self.file.store_path() / self.variant

    def cached(self) -> Path | None:
        return self.store.find(self.store_path())

    async def fetch(self) -> "Asset":
        with self.store.save(self.store_path(), self.get_mtime()) as f:
            f.write(await self.client.download(self.api_path()))
        return self


@final
class ChatsEntry(Timestamped, Nameable, Immutable):
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
        return self.chat_list.store_path() / "chats" / self.slug()

    def load_chat(self) -> Chat | None:
        return Chat._load(self.chat_list, store_path=self.chat_store_path())

    async def fetch_chat(self) -> Chat:
        return await Chat._fetch(self.chat_list, api_path=self.chat_api_path())

    def print(self) -> None:
        name = self.name or ""

        if sys.stdout.isatty():
            try:
                width = os.get_terminal_size().columns
            except OSError:
                width = 80

            max_len = width - 36 - 4
            if len(name) > max_len:
                name = name[: max_len - 1] + "…"

        print(f"{self.uuid}\t{name}")


@final
class Chats(APIObject):
    __slots__ = ("organization", "unseen", "_data")

    organization: "Organization"
    _data: dict[str, ChatsEntry]
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
        return self.organization.store_path()

    def set_data(self, data: Json) -> "Chats":
        # convert list (reverse chronological from API) to dict (forward chronological)
        entries = (
            ChatsEntry(self).set_data(raw) for raw in reversed(cast(list[JsonD], data))
        )
        self._data = {entry.uuid: entry for entry in entries}
        return self

    def get_data(self) -> Json:
        # convert dict (forward chronological) back to list (reverse chronological)
        return [entry._data for entry in reversed(self._data.values())]

    def entry(self, uuid: str) -> ChatsEntry | None:
        return self._data.get(uuid)

    def cached_entries(self) -> Iterator[ChatsEntry]:
        # yield in reverse chronological order (newest first)
        yield from reversed(self._data.values())

    async def new_entries(
        self, page_size: int = 20
    ) -> AsyncGenerator[ChatsEntry, None]:
        # "sliding window" sync (chat_conversations is recently-modified-first)
        new: dict[str, ChatsEntry] = {}

        if not self._data:
            # first fetch: grab everything in one unpaginated request (yes, the
            # api really does work that way, insanity), then check for mid-sync
            # changes by fetching the most recent chat, comparing with the sync
            self.set_data(await self.client.refresh(self.api_path()))
            for entry in self.cached_entries():
                yield entry
            self.save()
            offset = 0
            limit = 1
        else:
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
            for raw in page:
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
                    if entry.updated_at == new_entry.updated_at:
                        continue
                elif (
                    stored := self._data.get(uuid)
                ) and entry.updated_at == stored.updated_at:
                    # entry was in a chronological list of ones we already had,
                    # so we must also have all entries before it, so we're done
                    done = True
                    break

                new[uuid] = entry
                yield entry

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

    async def refresh(self) -> "Chats":
        async for _ in self.entries():
            pass

        return self

    def __len__(self) -> int:
        return len(self._data)


@final
class Organization(Nameable):
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

    def api_path(self) -> str:
        return f"organizations/{self.uuid}"

    def store_path(self) -> Path:
        return Path(self.account.slug()) / self.slug()

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
        return Organization(self.account).set_data(
            cast(JsonD, self._data["organization"])
        )


@final
class Account(Nameable):
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
        for membership in cast(list[JsonD], self._data["memberships"]):
            yield Membership(self).set_data(membership)

    def organization(self, uuid: str) -> Organization | None:
        for membership in self.memberships():
            organization = membership.organization()
            if organization.uuid == uuid:
                return organization

        return None
