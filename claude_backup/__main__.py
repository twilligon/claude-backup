# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitOverride=false, reportUnusedCallResult=false
# pyright: reportPrivateUsage=false, reportIncompatibleVariableOverride=false

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Iterable, Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import asyncio
import sys
from platformdirs import user_data_dir

from . import __version__
from .client import Client
from .store import Store
from .sync import Syncer

__all__ = (
    "main",
    "run",
)


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
    with suppress(KeyboardInterrupt):
        asyncio.run(_run(argv, keys))


def main() -> None:
    def keys() -> Iterator[str]:
        for key in filter(None, map(str.strip, sys.stdin)):
            if not key.startswith("sk-"):
                sys.exit(f"not a sessionKey: {key}")
            yield key

    run(sys.argv[1:], keys())


if __name__ == "__main__":
    main()
