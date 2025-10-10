#!/usr/bin/env python3
from contextlib import suppress
import json
import os
import time

import requests


class Client:
    RETRIES = 10
    MIN_RETRY_DELAY = 1
    MAX_RETRY_DELAY = 60
    SUCCESS_DELAY = 0.25

    def __init__(self, session_key: str):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Cookie": f"sessionKey={session_key}",
                "User-Agent": "Whig/0.1 (https://github.com/milkey-mouse/whig)",
            }
        )

    def get_cached(self, path: str) -> dict | list | None:
        with (
            suppress(FileNotFoundError, NotADirectoryError),
            open(f"responses/{path}.json", "r") as f,
        ):
            return json.load(f)

    def cache(self, path: str, data: dict | list) -> None:
        os.makedirs(os.path.dirname(f"responses/{path}.json"), exist_ok=True)
        with open(f"responses/{path}.json", "w") as f:
            json.dump(data, f)

    def refresh(self, path: str) -> dict | list:
        retry_delay = self.MIN_RETRY_DELAY

        for _ in range(self.RETRIES):
            r = self.session.get(f"https://claude.ai/api/{path}")
            if r.ok:
                data = r.json()
                self.cache(path, data)
                time.sleep(self.SUCCESS_DELAY)
                return data
            else:
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.MAX_RETRY_DELAY)

        r.raise_for_status()  # type: ignore
        raise requests.HTTPError  # pyright doesn't know above always throws

    def get(self, path: str) -> dict | list:
        return self.get_cached(path) or self.refresh(path)

    def get_changed(self, path: str, items, offset=0, limit=50, key=lambda o: o["uuid"]):
        new_items = {}
        moved = 0

        while True:
            if items:
                batch = self.refresh(f"{path}?limit={limit}&offset={offset}")
            else:
                batch = self.refresh(path)

            for item in batch:
                if items.get(key(item)) == item:
                    break
                elif new_items.get(key(item)) == item:
                    moved += 1
                else:
                    new_items[key(item)] = item
                    yield item
            else:
                offset += len(batch)
                limit *= 2 if items else 1
                if not items or len(batch) >= limit:
                    continue

            items.update(new_items)
            new_items.clear()

            if moved:
                # go back for items added while we refreshed (plus one item we
                # expect to have seen to probably limit this to one iteration)
                offset = 0
                limit = moved + 1
                continue

            break


def name(obj):
    return f"{obj["uuid"]} ({obj["name"]})" if obj.get("name") else obj["uuid"]


def main():
    with open("SESSION_KEY", "r") as f:
        session_key = f.read().strip()

    client = Client(session_key)

    for membership in client.get("account")["memberships"]:
        organization = membership["organization"]
        if "chat" not in organization["capabilities"]:
            print(f'Skipping organization {name(organization)} without "chat" capability')
            continue

        chat_conversations = f"organizations/{organization["uuid"]}/chat_conversations"
        chats = {chat["uuid"]: chat for chat in client.get_cached(chat_conversations) or ()}

        print(f"Fetching chats for organization {name(organization)}")

        # sliding window sync (chat_conversations is recently-modified-first)
        for chat in client.get_changed(chat_conversations, chats):
            print(name(chat))

            client.refresh(
                f"{chat_conversations}/{chat["uuid"]}"
                "?tree=True&rendering_mode=messages&render_all_tools=true"  # TODO: try rendering_mode=raw
            )

        client.cache(chat_conversations, list(chats.values()))


if __name__ == "__main__":
    main()