"""End-to-end sync tests against the in-memory FakeAPI from conftest.py."""

import json


async def test_initial_sync_populates_cache(api, sync, chats_list_path, chats_dir):
    """First sync downloads every chat and writes the chats list."""
    for i in range(3):
        api.chats_list.insert(0, api.make_chat(f"C{i}", f"2026-01-0{i + 1}T00:00:00Z"))

    await sync()

    saved = json.loads(chats_list_path.read_text())
    assert {e["name"] for e in saved} == {"C0", "C1", "C2"}
    assert len(list(chats_dir.glob("*.json"))) == 3


async def test_no_change_resync_is_noop(api, sync):
    """A second sync with no API changes triggers no chat-detail fetches and
    just one list call (the sliding-window's first-page probe)."""
    api.chats_list.insert(0, api.make_chat("A", "2026-01-01T00:00:00Z"))
    api.chats_list.insert(0, api.make_chat("B", "2026-01-02T00:00:00Z"))

    await sync()
    api.chat_call_count = 0
    api.list_call_count = 0

    await sync()

    assert api.chat_call_count == 0
    assert api.list_call_count == 1


async def test_updated_chat_is_refetched(api, sync, chats_list_path, chats_dir):
    """An entry whose `updated_at` changed gets re-fetched; others don't."""
    a = api.make_chat("A", "2026-01-01T00:00:00Z")
    b = api.make_chat("B", "2026-01-02T00:00:00Z")
    api.chats_list = [b, a]

    await sync()

    a["updated_at"] = "2026-02-01T00:00:00Z"
    api.chat_details[a["uuid"]]["updated_at"] = a["updated_at"]
    api.chat_details[a["uuid"]]["chat_messages"].append({"text": "new!"})
    api.chats_list = [a, b]
    api.chat_call_count = 0

    await sync()

    assert api.chat_call_count == 1
    saved = json.loads(chats_list_path.read_text())
    assert next(e for e in saved if e["name"] == "A")["updated_at"] == a["updated_at"]
    detail_path = next(chats_dir.glob("A-*.json"))
    assert len(json.loads(detail_path.read_text())["chat_messages"]) == 2


async def test_galaxy_brain_mid_sync_insert(api, sync, chats_list_path, chats_dir):
    """A new chat appearing between two pages of new_entries() is still caught.

    Setup: cached state contains a single old chat E. The API then gains 22 new
    chats A21..A00 on top of E, so page 1 (limit=20) doesn't reach E. After
    list call 1 returns, insert NEW at offset 0, which shifts everything down
    by one. The dup-detection on page 2's A02 bumps `self.unseen` to 1, the
    outer entries() loop calls new_entries() a second time with `limit=2`, and
    NEW is caught at offset 0."""
    e = api.make_chat("E", "2026-01-01T00:00:00Z")
    api.chats_list = [e]
    await sync()

    new_top = [
        api.make_chat(f"A{i:02}", f"2026-02-01T00:{i:02}:00Z") for i in range(22)
    ][::-1]
    api.chats_list = new_top + [e]
    new_chat = api.make_chat("NEW", "2026-03-01T00:00:00Z")
    api.list_call_count = 0  # reset so mutation timing is relative to sync 2
    api.mid_sync_mutations = [(1, lambda a: a.chats_list.insert(0, new_chat))]

    await sync()

    saved = json.loads(chats_list_path.read_text())
    expected = {entry["uuid"] for entry in [*new_top, e, new_chat]}
    assert {entry["uuid"] for entry in saved} == expected
    assert len(list(chats_dir.glob("*.json"))) == 24


async def test_missing_detail_file_is_refetched(api, sync, chats_dir):
    """A chats-list entry whose detail file is missing on disk gets re-fetched
    on the next sync. This is what recovers an interrupted sync that wrote the
    list but didn't finish all detail fetches."""
    a = api.make_chat("A", "2026-01-01T00:00:00Z")
    b = api.make_chat("B", "2026-01-02T00:00:00Z")
    api.chats_list = [b, a]

    await sync()
    a_path = next(chats_dir.glob("A-*.json"))
    a_path.unlink()
    api.chat_call_count = 0

    await sync()

    assert api.chat_call_count == 1
    assert a_path.exists()
