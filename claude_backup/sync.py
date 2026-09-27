# pyright: reportAny=false, reportExplicitAny=false
# pyright: reportImplicitOverride=false, reportUnusedCallResult=false
# pyright: reportPrivateUsage=false, reportIncompatibleVariableOverride=false

from collections.abc import AsyncGenerator, Awaitable
from contextlib import aclosing
from dataclasses import dataclass
import sys

from .client import Client
from .store import Account, Chat, ChatsEntry, File, Organization, Store
from .util import as_completed


@dataclass(slots=True)
class Syncer:
    client: Client
    store: Store
    connections: int = 6
    success_delay: float = 0.25

    async def get_organizations(self) -> AsyncGenerator[Organization]:
        account = Account(self)
        account.set_data(await self.client.refresh(account.api_path()))

        old_account = next(
            (old for old in Account.load(self) if old.uuid == account.uuid),
            None,
        )
        if old_account and old_account.slug() != account.slug():
            print(
                f"Renaming account {old_account} to {account.name or account.uuid}",
                file=sys.stderr,
            )
            account.rename_cached(old_account)
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
                organization.rename_cached(old_organization)

            if "chat" not in organization.capabilities:
                print(
                    f'Skipping organization {organization} without "chat" capability',
                    file=sys.stderr,
                )
                continue

            print(f"Fetching chats for organization {organization}", file=sys.stderr)
            yield organization

    async def fetches(self) -> AsyncGenerator[Awaitable[Chat | File], None]:
        async def fetch_new_chat(entry: ChatsEntry, old_chat: Chat | None) -> Chat:
            chat = await entry.fetch_chat()

            if old_chat and old_chat.store_path() != chat.store_path():
                old_chat.delete_cached()

            return chat

        async for organization in self.get_organizations():
            async for entry in organization.chat_list().entries():
                # we must get old_entry *now* and not in the async function in
                # fetch_new_chat, since Chats.new_entries might finish and save
                # the new entry over old_entry!
                old_entry = entry.chat_list.entry(entry.uuid)
                old_chat = old_entry.load_chat() if old_entry else None
                if old_chat and old_chat.updated_at == entry.updated_at:
                    continue

                entry.print()
                yield fetch_new_chat(entry, old_chat)

            async for file in organization.file_list().entries():
                if file.cached() is None:
                    yield file.fetch()

    async def sync_all(self) -> None:
        async with aclosing(
            as_completed(self.fetches(), self.connections, self.success_delay)
        ) as tasks:
            async for task in tasks:
                await task
