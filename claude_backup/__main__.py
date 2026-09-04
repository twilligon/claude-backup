# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitOverride=false, reportUnusedCallResult=false
# pyright: reportPrivateUsage=false, reportIncompatibleVariableOverride=false

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import asyncio
import os
import sys
from platformdirs import user_data_dir
import browser_cookie3  # pyright: ignore[reportMissingTypeStubs]

from . import __version__
from .client import Client
from .store import Store
from .sync import Syncer

__all__ = (
    "main",
    "run",
    "get_session_key",
)


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


async def _run(argv: Sequence[str]) -> None:
    def default(cls: type[Any], key: str) -> Any:
        return cls.__dataclass_fields__[key].default

    parser = ArgumentParser(
        prog="claude-backup",
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

    args = parser.parse_args(argv)
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


def run(argv: Sequence[str]) -> None:
    with suppress(KeyboardInterrupt):
        asyncio.run(_run(argv))


def main() -> None:
    run(sys.argv[1:])


if __name__ == "__main__":
    main()
