# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitOverride=false, reportUnusedCallResult=false
# pyright: reportPrivateUsage=false, reportIncompatibleVariableOverride=false

from asyncio import Task
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Iterable
from contextlib import aclosing, suppress
from dataclasses import dataclass
from typing import TypeVar
import asyncio
import sys

from .client import Client
from .store import Account, Chat, Chats, ChatsEntry, Organization, Store

__all__ = ("Syncer",)


T = TypeVar("T")


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
    success_delay: float = 0.25

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

    async def new_chat_fetches(self) -> AsyncGenerator[Task[Chat], None]:
        def fetch_new_chat(entry: ChatsEntry, old_chat: Chat | None) -> Task[Chat]:
            async def _fetch_new_chat() -> Chat:
                chat = await entry.fetch_chat()

                if old_chat and old_chat.store_path() != chat.store_path():
                    old_chat.delete_cached()

                return chat

            return asyncio.create_task(_fetch_new_chat())

        chats_lists: list[Chats] = [
            organization.chat_list() async for organization in self.get_organizations()
        ]

        async with aclosing(
            aroundrobin(*(chats.entries() for chats in chats_lists))
        ) as items:
            async for entry in items:
                # we must get old_entry *now* and not in the async function in
                # fetch_new_chat, since Chats.new_entries might finish and save
                # the new entry over old_entry!
                old_entry = entry.chat_list.entry(entry.uuid)
                old_chat = old_entry.load_chat() if old_entry else None
                if old_chat and old_chat.updated_at == entry.updated_at:
                    continue

                entry.print()
                yield fetch_new_chat(entry, old_chat)

    async def sync_all(self) -> None:
        async with aclosing(self.as_completed(self.new_chat_fetches())) as tasks:
            async for task in tasks:
                await task
