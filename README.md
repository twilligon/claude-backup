# `claude-backup`

Unofficial, unsanctioned tool to backup [Claude.ai](https://claude.ai) chats to local files.


## Features

`claude-backup` creates a full local copy of all (text) content of all branches of all chats accessible to your [Claude.ai](https://claude.ai) account. If you are a member of multiple "organizations", it fetches chats from all of them. We preserve all metadata on chats and their provenance, including that of their parent organization and user. We automatically rename our local copies of chats and organizations to match their current names on `claude.ai`.

Chats are stored as their original API JSON with nice `find`able names and `grep`able contents:

```
$ CLAUDE_SESSION_KEY="$(wl-paste)" uvx claude-backup
$ tree ~/.local/share/claude-backup | head -n15
~/.local/share/claude-backup
├── account.json
├── organizations
│   └── claude@twilligon.com's_Organization-9e9a56fc-6d1c-4d62-a96d-0cff3a473cf0
│       ├── chat_conversations
│       │   ├── Free_Indirect_Discourse_Analysis-48d70be1-f23d-4757-9bcf-9d9d9711a3f6.json
│       │   ├── Fetty_Wap_name_origin-dc45880a-8c60-4366-b610-e4fe6bb9a65c.json
│       │   ├── Ryzen_Motherboard_PS_2_Port_Hunt-012583f0-83cc-4c18-8f8b-312c8cb856ac.json
│       │   ├── Tailscale_versus_wireguard_comparison-24ed7155-8d9e-410f-bf7c-cd3fb6cb4379.json
│       │   ├── Credit_card_companies_in_Europe-67ad5f75-80ba-4754-bc6c-7d21e98f948c.json
│       │   ├── OpenGL_and_Vulkan_Package_Compatibility-63859641-5ffe-459b-99b0-4759acdc8235.json
│       │   ├── Government_tech_capabilities_and_bureaucracy-1c0743f0-a124-456b-888f-d08c5b83923b.json
│       │   ├── Reverse_Engineering_Minified_JavaScript-8cf7cab1-9576-45d7-929a-ea34997b0061.json
│       │   ├── Wire_and_String_Mysteries-9ac747b7-f36e-4e86-9f24-377ab16fb5ec.json
│       │   ├── US_Senate_parliamentarian's_role_and_power-5fa34eb3-ac74-4683-8850-c90761e5f3d8.json
$ rg -0l falafel ~/.local/share/claude-backup/organizations/*/chat_conversations | xargs -0n1 basename
Fixing_dry_falafel_centers-08217feb-c912-40c2-9d14-f89784d61ab5.json
Best_falafel_restaurants_in_San_Francisco-76aa159a-e41a-4935-a2e5-d53465041e13.json
Rainy_day_food_delivery_dilemma-db4992e8-c69c-4085-a571-9a22dfff68da.json
Pita_Chip_Conversation_Search-059dca9c-bac3-4cd5-b405-34ddbbdd5fa8.json
$ rg -0l falafel ~/.local/share/claude-backup/organizations/*/chat_conversations | xargs -0 jq -r '"https://claude.ai/chat/\(.uuid)"'
https://claude.ai/chat/08217feb-c912-40c2-9d14-f89784d61ab5
https://claude.ai/chat/76aa159a-e41a-4935-a2e5-d53465041e13
https://claude.ai/chat/db4992e8-c69c-4085-a571-9a22dfff68da
https://claude.ai/chat/059dca9c-bac3-4cd5-b405-34ddbbdd5fa8
```

We use an incremental sync algorithm to fetch only chats created or updated since the last backup. The fetch is done in parallel, with a typical user agent and polite rate and connection limits such that it's less traffic than manually scrolling through your chats and opening each in a new browser tab. I of course can't guarantee Big Claude won't be after you if you run this unofficial tool, but empirically they don't seem to mind.

## Limitations

This read-only tool is the product of reverse-engineering Claude.ai's internal API (for Good, not Evil---please don't ban me Anthropic 🙏) so I can't make any guarantees `claude-backup` will continue to work. That said, we make very few assumptions about the API schema, and everything works as of 2025-10-31. I'll likely update this best-effort when things break. Barring that, PRs welcome ;)

For reliable and comprehensive backups even through minor API changes, we save raw JSON responses from the API instead of normalizing them to some fixed schema. This should be a bit more resilient than forcing everything into some internal data model that could diverge from that of Claude.ai, but it means we don't download resources referenced by API objects other than what's necessary to list and fetch chat text. In practice, this preserves all textual message content and text/markdown attachments, but not images, PDFs, container uploads, "advanced research" reports, etc.

Because of how our incremental sync works, if `claude-backup` is interrupted, the next sync must start from scratch. This is moderately easy to fix but I'm lazy :)

As a backup tool, we *retain deleted chats* by default. To start from scratch and delete our local copies of deleted chats, manually delete the `backup_dir` or run `claude-backup --ignore-cache`.

By default, `claude-backup` attempts to authenticate to `claude.ai` by extracting a session cookie from your browser. If this doesn't work (and frankly if this does work you should be sandboxing things better!) you must manually do the same. For Chrome et al., to https://claude.ai in your browser, open *Developer tools* with F12 or Ctrl+Shift+I, navigate to the *Application* tab (it may be hidden under *⋮* > *More tools* > *Application*), and copy the value of the `sessionKey` cookie. (Firefox should be [similar](https://firefox-source-docs.mozilla.org/devtools-user/storage_inspector/index.html).) Then set the `CLAUDE_SESSION_KEY` environment variable to this cookie when running `claude-backup`:

    $ CLAUDE_SESSION_KEY="sk-ant-sid01-..." claude-backup

As this tool demonstrates *anyone with this cookie is authenticated as you on `claude.ai`* so be careful and never give this to anyone or anything you do not trust! It might even be worth keeping out of `history` by loading it straight from your clipboard with e.g. `CLAUDE_SESSION_KEY="$(wl-paste)"`, though the exact command varies by platform. I'm sure Claude knows which you should use ;)

## Install

The blazing fast and memory safe way:

    $ uvx claude-backup  # installed on demand

The traditional way:

    $ pip install claude-backup
    $ claude-backup

The bleeding-edge way:

    $ git clone https://github.com/twilligon/claude-backup
    $ cd claude-backup
    $ python3 -m venv venv; . venv/bin/activate  # you probably want a venv
    $ pip install -e .
    $ claude-backup

## Usage_

    $ claude-backup --help
    usage: claude-backup [-h] [-v] [-c CONNECTIONS] [-s DELAY] [-r RETRIES]
                         [--min-retry-delay DELAY] [--max-retry-delay DELAY]
                         [--ignore-cache]
                         [backup_dir]

    Backup Claude.ai chats

    positional arguments:
      backup_dir            Directory to save backups (default:
                            ~/.local/share/claude-backup)

    options:
      -h, --help            show this help message and exit
      -v, --version         show program's version number and exit
      -c, --connections CONNECTIONS
                            Maximum concurrent connections (default: 6)
      -s, --success-delay DELAY
                            Delay after successful request in seconds (default:
                            0.1)
      -r, --retries RETRIES
                            Number of retries for API requests (default: 10)
      --min-retry-delay DELAY
                            Minimum retry delay in seconds (default: 1.0)
      --max-retry-delay DELAY
                            Maximum retry delay in seconds (default: 60.0)
      --ignore-cache        Ignore local cache and re-fetch everything from API
                            (default: False)

## Authentication

Set the `CLAUDE_SESSION_KEY` environment variable to your claude.ai sessionKey cookie. If not set, `claude-backup` will attempt to load cookies from your browser automatically.

## License

`claude-backup` is dedicated to the public domain where possible via CC0-1.0.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
