"""Shared fixtures: an in-memory mock of claude.ai's API surface and helpers for
driving a sync against it."""

from collections.abc import Callable
from itertools import count
from typing import Any

import pytest

from claude_backup import Client, Store, Syncer

ACCOUNT_UUID = "00000000-0000-0000-0000-000000000001"
ORG_UUID = "00000000-0000-0000-0000-00000000000a"


class FakeAPI:
    """Minimal in-memory mock of claude.ai's API surface, just enough for sync."""

    def __init__(self) -> None:
        self.account_data: dict[str, Any] = {
            "uuid": ACCOUNT_UUID,
            "memberships": [
                {
                    "organization": {
                        "uuid": ORG_UUID,
                        "name": "org",
                        "capabilities": ["chat"],
                    }
                }
            ],
        }
        self.org_data: dict[str, Any] = {
            "uuid": ORG_UUID,
            "name": "org",
            "capabilities": ["chat"],
        }
        # `chats_list` is newest-first, matching the real API.
        self.chats_list: list[dict[str, Any]] = []
        self.chat_details: dict[str, dict[str, Any]] = {}
        self.list_call_count = 0
        self.chat_call_count = 0
        # Optional mid-sync mutations: each entry is (after_call_n, fn(api)). The
        # mutation fires after the Nth list-endpoint call has returned its
        # response, letting tests simulate concurrent activity on claude.ai.
        self.mid_sync_mutations: list[tuple[int, Callable[["FakeAPI"], None]]] = []
        self._uuid_counter = count(0x100)

    def make_chat(self, name: str, updated_at: str) -> dict[str, Any]:
        """Create a new chat with a unique uuid. Returns the list-entry dict; the
        corresponding chat-detail dict is stored in `self.chat_details` and only
        served from the `/chat_conversations/<uuid>` endpoint."""
        n = next(self._uuid_counter)
        uuid = f"00000000-0000-0000-0000-{n:012x}"
        entry = {
            "uuid": uuid,
            "name": name,
            "updated_at": updated_at,
            "created_at": updated_at,
        }
        self.chat_details[uuid] = {**entry, "chat_messages": [{"text": f"body-{name}"}]}
        return entry

    def respond(self, path: str) -> Any:
        if path == "account":
            return self.account_data
        if path == f"organizations/{ORG_UUID}":
            return self.org_data
        if path.startswith(f"organizations/{ORG_UUID}/chat_conversations"):
            after = path.split("chat_conversations", 1)[1]
            if after.startswith("/"):
                uuid = after[1:].split("?")[0]
                self.chat_call_count += 1
                return self.chat_details[uuid]
            query = after[1:] if after.startswith("?") else ""
            params = dict(p.split("=") for p in query.split("&")) if query else {}
            limit = int(params.get("limit", 999_999))
            offset = int(params.get("offset", 0))
            self.list_call_count += 1
            response = self.chats_list[offset : offset + limit]
            for n, mutate in list(self.mid_sync_mutations):
                if self.list_call_count == n:
                    mutate(self)
                    self.mid_sync_mutations.remove((n, mutate))
            return response
        raise RuntimeError(f"unexpected path: {path}")


@pytest.fixture
def api() -> FakeAPI:
    return FakeAPI()


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(store_dir=tmp_path)


@pytest.fixture
def sync(api: FakeAPI, store: Store, monkeypatch) -> Callable[[], Any]:
    """Returns an async callable that runs one full sync against `api` and
    `store`. Patches `Client.refresh` to dispatch into the mock instead of
    talking to claude.ai."""

    async def fake_refresh(_self: Client, path: str) -> Any:
        return api.respond(path)

    monkeypatch.setattr(Client, "refresh", fake_refresh)

    async def run() -> None:
        async with Client(session_key="fake") as client:
            await Syncer(client=client, store=store, success_delay=0).sync_all()

    return run


@pytest.fixture
def org_dir(store: Store):
    return store.store_dir / "organizations" / f"org-{ORG_UUID}"


@pytest.fixture
def chats_list_path(org_dir):
    return org_dir / "chat_conversations.json"


@pytest.fixture
def chats_dir(org_dir):
    return org_dir / "chat_conversations"
