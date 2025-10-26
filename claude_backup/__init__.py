"""Backup Claude.ai chats to local JSON files"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from tempfile import TemporaryDirectory
from typing import Any, Literal, TypedDict
from uuid import UUID
import json
import os
import shutil
import time

from fake_useragent import UserAgent
from platformdirs import user_data_dir
from typeguard import check_type
import browser_cookie3
import requests

# TODO: compare file mtimes for updates instead of using json list (ehh)
# TODO: factor "cache" (which is really anything but) out of Client

__version__ = "0.1.4"

__all__ = ("__version__", "main", "Client")


class Organization(TypedDict):
    uuid: str
    name: str
    capabilities: list[str]


class Membership(TypedDict):
    organization: Organization


class Account(TypedDict):
    memberships: list[Membership]


MIGRATIONS = defaultdict(lambda: (__version__, lambda old, new: None))


@contextmanager
def umask(mask: int):
    old_mask = os.umask(mask)
    try:
        yield
    finally:
        _ = os.umask(old_mask)


def migrate(backup_dir: str) -> None:
    if not os.path.exists(backup_dir):
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

    with open(os.path.join(backup_dir, "version")) as f:
        print(__version__, file=f)


def initialize_cache(backup_dir: str) -> None:
    version_file = os.path.join(backup_dir, "version")

    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir, exist_ok=True)
        with open(version_file, "w") as f:
            f.write(__version__)
        return

    try:
        with open(version_file) as f:
            current_version = f.read().strip()
    except FileNotFoundError:
        current_version = None

    if current_version == __version__:
        return

    migrate(backup_dir)


def uuid(obj: Mapping[str, Any]) -> str:
    return str(UUID(obj["uuid"]))


def name(obj: Mapping[str, Any]) -> str:
    return f"{uuid(obj)} ({obj['name']})" if obj.get("name") else uuid(obj)


def cache(*parts: Mapping[str, Any]) -> str:
    SANITIZE_FILENAME = {ord(c): ord("_") for c in '<>:"|?*/\\ \t\n\r'}
    name = lambda p: p.get("name", "").translate(SANITIZE_FILENAME)
    return os.path.join(
        *(f"{name(p)}-{uuid(p)}" if name(p) else uuid(p) for p in parts)
    )


@dataclass(slots=True)
class Client:
    backup_dir: str
    retries: int = 10
    success_delay: float = 0.25
    min_retry_delay: float = 1
    max_retry_delay: float = 60
    force_refresh: bool = False
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        self.session.headers["User-Agent"] = UserAgent().chrome
        if session_key := os.environ.get("CLAUDE_SESSION_KEY"):
            self.session.cookies.set("sessionKey", session_key, domain=".claude.ai")
        else:
            try:
                self.session.cookies = (
                    browser_cookie3.load()
                )  # pyright: ignore[reportAttributeAccessIssue]
            except Exception as e:
                raise RuntimeError(
                    "Failed to load browser cookies. "
                    "Set CLAUDE_SESSION_KEY to your claude.ai sessionKey cookie."
                ) from e

    def cache(
        self, cache: str | Literal[False], data: dict[Any, Any] | list[Any]
    ) -> None:
        if cache:
            cache_file = os.path.join(self.backup_dir, f"{cache}.json")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(data, f)

    def get_cached(
        self, cache: str | Literal[False]
    ) -> dict[Any, Any] | list[Any] | None:
        if cache and not self.force_refresh:
            cache_file = os.path.join(self.backup_dir, f"{cache}.json")
            with suppress(FileNotFoundError, NotADirectoryError), open(cache_file) as f:
                return json.load(f)

    def refresh(
        self, path: str, cache: str | bool = False
    ) -> dict[Any, Any] | list[Any]:
        cache = path if cache is True else cache

        retry_delay = self.min_retry_delay
        r = None
        for _ in range(self.retries):
            r = self.session.get(f"https://claude.ai/api/{path}", allow_redirects=False)
            if r.ok:
                data = r.json()
                self.cache(cache, data)
                time.sleep(self.success_delay)
                return data
            else:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)

        assert r is not None
        r.raise_for_status()
        raise requests.HTTPError  # pyright doesn't know above always throws

    def get(self, path: str, cache: str | bool = True) -> dict[Any, Any] | list[Any]:
        cache = path if cache is True else cache
        return self.get_cached(cache) or self.refresh(path, cache)

    def get_paginated(
        self,
        path: str,
        cache: str,
        items: dict[str, Any],
        offset: int = 0,
        limit: int = 20,
        key: Callable[[dict[str, Any]], str] = uuid,
    ) -> Generator[dict[str, Any], None, None]:
        batch = {}

        if not items:
            for item in self.refresh(path):
                batch[key(item)] = item
                yield item

            items.update(batch)
            batch.clear()

        new_items = 0
        while True:
            page = self.refresh(f"{path}?limit={limit}&offset={offset}")
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
                yield item

            # we *ought* to break from this loop by seeing something from a
            # prior batch, but just in case e.g. all items in `items` have
            # since been deleted (or moved/are counted in new_items), we're
            # also definitely done if they ran out of items for this page
            if (
                len(page) >= limit and key(item) not in items
            ):  # pyright: ignore[reportPossiblyUnboundVariable]
                offset += limit
                limit *= 2
                continue
            else:
                items.update(batch)
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


@dataclass(frozen=True, slots=True)
class DefaultPath:
    path: str

    def __str__(self) -> str:
        home = os.path.expanduser("~")
        if self.path.startswith(home):
            return "~" + self.path[len(home) :]
        return self.path


def main() -> None:
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
        "-r",
        "--retries",
        type=int,
        default=default("retries"),
        help="Number of retries for API requests",
    )
    parser.add_argument(
        "-s",
        "--success-delay",
        type=float,
        default=default("success_delay"),
        help="Delay after successful request in seconds",
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

    client = Client(**vars(args))

    account = check_type(client.get("account"), Account)
    for membership in account.get("memberships", []):
        org = membership["organization"]
        if "chat" not in org["capabilities"]:
            print(f'Skipping organization {name(org)} without "chat" capability')
            continue
        else:
            print(f"Fetching chats for organization {name(org)}")

        chats = {chat["uuid"]: chat for chat in client.get_cached(cache(org)) or ()}

        # sliding window sync (chat_conversations is recently-modified-first)
        for chat in client.get_paginated(
            f"organizations/{uuid(org)}/chat_conversations", cache(org), chats
        ):
            print(name(chat))

            if old := chats.get(uuid(chat)):
                with suppress(FileNotFoundError, OSError):
                    os.rename(
                        os.path.join(client.backup_dir, f"{cache(org, old)}.json"),
                        os.path.join(client.backup_dir, f"{cache(org, chat)}.json"),
                    )

            client.refresh(
                f"organizations/{uuid(org)}/chat_conversations/{uuid(chat)}"
                f"?tree=True&rendering_mode=messages&render_all_tools=true",  # TODO: try rendering_mode=raw ?
                cache(org, chat),
            )

        # NOTE: cached chats list is unsorted! but this is fine for us
        client.cache(cache(org), list(chats.values()))


if __name__ == "__main__":
    main()
