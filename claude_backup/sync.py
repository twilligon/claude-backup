# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitOverride=false, reportUnusedCallResult=false
# pyright: reportPrivateUsage=false, reportIncompatibleVariableOverride=false

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass
from typing import TypeAlias
import sys

from .client import Client
from .store import Account, Asset, Chat, ChatsEntry, Organization, Store
from .util import Channel, amerge, as_completed

__all__ = (
    "Syncer",
    "Fetch",
)


Fetch: TypeAlias = Awaitable[Chat | Asset]


@dataclass(slots=True)
class Syncer:
    client: Client
    store: Store
    connections: int = 6
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
        if old_account and old_account.store_path() != account.store_path():
            print(
                f"Renaming account {old_account} to {account.name or account.uuid}",
                file=sys.stderr,
            )
            self.store.rename(old_account.store_path(), account.store_path())
            self.store.delete(old_account.store_path())
        account.save()

        for membership in account.memberships():
            organization = membership.organization()
            store_path = organization.store_path()

            if (
                old_account
                and (old_organization := old_account.organization(organization.uuid))
                and old_organization.slug() != organization.slug()
            ):
                print(
                    f"Renaming organization {old_organization} to {organization.name or organization.uuid}",
                    file=sys.stderr,
                )
                self.store.rename(
                    store_path.with_name(old_organization.slug()), store_path
                )
                self.store.rename(
                    store_path.with_name(old_organization.slug() + ".json"),
                    store_path.with_name(store_path.name + ".json"),
                )

            if "chat" not in organization.capabilities:
                print(
                    f'Skipping organization {organization} without "chat" capability',
                    file=sys.stderr,
                )
                continue

            print(f"Fetching chats for organization {organization}", file=sys.stderr)
            yield organization

    async def new_fetches(self, channel: Channel[Fetch]) -> AsyncGenerator[Fetch, None]:
        async def queue_new_assets(
            put: Callable[[Fetch], Awaitable[None]], chat: Chat
        ) -> None:
            for file in chat.files():
                for asset in file.assets():
                    if asset.cached() is None:
                        await put(asset.fetch())

        async def fetch_new_chat(entry: ChatsEntry, old_chat: Chat | None) -> Chat:
            async with channel.writer() as put:
                chat = await entry.fetch_chat()

                if old_chat and old_chat.store_path() != chat.store_path():
                    old_chat.delete_cached()

                await queue_new_assets(put, chat)

                return chat

        async with channel.writer() as put, aclosing(
            amerge(
                *[
                    organization.chat_list().entries()
                    async for organization in self.get_organizations()
                ]
            )
        ) as items:
            async for entry in items:
                # we must get old_entry *now* and not in the async function in
                # fetch_new_chat, since Chats.new_entries might finish and save
                # the new entry over old_entry!
                old_entry = entry.chat_list.entry(entry.uuid)
                old_chat = old_entry.load_chat() if old_entry else None
                if old_chat and old_chat.updated_at == entry.updated_at:
                    await queue_new_assets(put, old_chat)
                    continue

                entry.print()
                yield fetch_new_chat(entry, old_chat)

    async def sync_all(self) -> None:
        channel: Channel[Fetch] = Channel()
        async with channel.reader() as reader, aclosing(
            as_completed(
                amerge(reader, self.new_fetches(channel)),
                self.connections,
                self.success_delay,
            )
        ) as tasks:
            async for task in tasks:
                await task
