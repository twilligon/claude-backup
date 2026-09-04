# `claude-backup`

Unofficial, unsanctioned tool to back up [Claude.ai](https://claude.ai) chats to local files.

    uvx claude-backup

## Features

`claude-backup` creates a full local copy of all content of all branches of all chats accessible to your [Claude.ai](https://claude.ai) account. If you are a member of multiple "organizations", it fetches chats from all of them. We preserve all metadata on chats and their provenance, including that of their parent organization and account. We automatically rename our local copies of accounts, organizations, and chats to match their current names on Claude.ai.

We use an incremental sync algorithm to fetch only chats created or updated since the last backup. The fetch is done in parallel, with a typical user agent and polite rate and connection limits such that it's less traffic than manually scrolling through your chats and opening each in a new browser tab. I of course can't guarantee Big Claude won't be after you if you run this unofficial tool, but empirically they don't seem to mind.

To convert the saved JSON to Markdown (e.g. for feeding chats back to a model), see the companion tool [`claude2md`](https://github.com/twilligon/claude2md).

Chats are stored as their original API JSON with nice `find`able names and `grep`able contents:

    $ wl-paste | uvx claude-backup
    Skipping organization twilligon (c8eaca6b-eddb-4bbc-9fe3-637a0574565f) without "chat" capability
    Fetching chats for organization claude@twilligon.com's Organization (9e9a56fc-6d1c-4d62-a96d-0cff3a473cf0)
    30ceebd6-afcc-4796-bb77-631069cd0696    Cadre versus posse comparison
    d15ac7d6-f167-43d3-afac-fcfaaafb64ec    GraphQL database backends and data sources
    3e553492-89a8-45e8-b51a-b2faf9dfde64    Japanese sword quality despite poor iron sources
    040343e8-db5a-4031-a21a-be237a6c661a    Balisong pin and screw maintenance
    08217feb-c912-40c2-9d14-f89784d61ab5    Fixing dry falafel centers
    d41bf95e-8da1-47b3-af89-aecacd7770a6    Sleep Token's artistic merits
    b10e02dd-479b-4714-9d99-264d6ef4ba75    Annual human steel consumption
    608f7e3d-fbcc-4891-bf67-ebc9f15c7a01    Calling payphones from mobile devices
    c56ad0c5-feae-4dae-bafc-784d1f46de29    Postfix security assessment
    4553fae5-61a6-40fc-b95c-36c53e706614    App store review power dynamics for major companies
    0b7f4ae8-ef57-4cac-8d47-7e4c5b0e5564    Exercise timing and sleep quality
    6e5385c4-6ad4-4d46-ac20-dda2105a3bea    Scented neural networks with odor emissions
    $ # ...and so on... time passes... then later:
    $ wl-paste | uvx claude-backup
    Skipping organization twilligon (c8eaca6b-eddb-4bbc-9fe3-637a0574565f) without "chat" capability
    Fetching chats for organization claude@twilligon.com's Organization (9e9a56fc-6d1c-4d62-a96d-0cff3a473cf0)
    4ddbcc12-5cd8-4611-813a-befdedeb4b16    Smithsonian funding and government ownership
    $ tree ~/.local/share/claude-backup | head -n15
    ~/.local/share/claude-backup
    └── 0.1.13
        ├── claude@twilligon.com-f6c59c68-2f18-4ec0-b488-b237e1f82045
        │   ├── claude@twilligon.com's_Organization-9e9a56fc-6d1c-4d62-a96d-0cff3a473cf0
        │   │   ├── chats
        │   │   │   ├── Free_Indirect_Discourse_Analysis-48d70be1-f23d-4757-9bcf-9d9d9711a3f6.json
        │   │   │   ├── Fetty_Wap_name_origin-dc45880a-8c60-4366-b610-e4fe6bb9a65c.json
        │   │   │   ├── Ryzen_Motherboard_PS_2_Port_Hunt-012583f0-83cc-4c18-8f8b-312c8cb856ac.json
        │   │   │   ├── Tailscale_versus_wireguard_comparison-24ed7155-8d9e-410f-bf7c-cd3fb6cb4379.json
        │   │   │   ├── Credit_card_companies_in_Europe-67ad5f75-80ba-4754-bc6c-7d21e98f948c.json
        │   │   │   ├── OpenGL_and_Vulkan_Package_Compatibility-63859641-5ffe-459b-99b0-4759acdc8235.json
        │   │   │   ├── Government_tech_capabilities_and_bureaucracy-1c0743f0-a124-456b-888f-d08c5b83923b.json
        │   │   │   ├── Reverse_Engineering_Minified_JavaScript-8cf7cab1-9576-45d7-929a-ea34997b0061.json
        │   │   │   ├── Wire_and_String_Mysteries-9ac747b7-f36e-4e86-9f24-377ab16fb5ec.json
        │   │   │   ├── US_Senate_parliamentarian's_role_and_power-5fa34eb3-ac74-4683-8850-c90761e5f3d8.json
    $ rg -0l falafel ~/.local/share/claude-backup/*/*/*/chats | xargs -0n1 basename
    Fixing_dry_falafel_centers-08217feb-c912-40c2-9d14-f89784d61ab5.json
    Best_falafel_restaurants_in_San_Francisco-76aa159a-e41a-4935-a2e5-d53465041e13.json
    Rainy_day_food_delivery_dilemma-db4992e8-c69c-4085-a571-9a22dfff68da.json
    Pita_Chip_Conversation_Search-059dca9c-bac3-4cd5-b405-34ddbbdd5fa8.json
    $ rg -0l falafel ~/.local/share/claude-backup/*/*/*/chats | xargs -0 jq -r '"https://claude.ai/chat/\(.uuid)"'
    https://claude.ai/chat/08217feb-c912-40c2-9d14-f89784d61ab5
    https://claude.ai/chat/76aa159a-e41a-4935-a2e5-d53465041e13
    https://claude.ai/chat/db4992e8-c69c-4085-a571-9a22dfff68da
    https://claude.ai/chat/059dca9c-bac3-4cd5-b405-34ddbbdd5fa8

## Limitations

This read-only tool is the product of reverse-engineering Claude.ai's internal API (for Good, not Evil---please don't ban me Anthropic 🙏) so I can't make any guarantees `claude-backup` will continue to work. That said, we make few assumptions about the API schema, and everything works as of 2026-09-04. I'll likely update this best-effort when things break. Barring that, PRs welcome ;)

For reliable and comprehensive backups even through minor API changes, we save raw JSON responses from the API instead of normalizing them to some fixed schema. This should be a bit more resilient than forcing everything into some internal data model that could diverge from that of Claude.ai, but it means we don't necessarily download all resources referenced by API objects other than what's necessary to list and fetch chat content. In practice, this preserves all regular chat content, but not necessarily container uploads, "advanced research" reports, etc.

As a backup tool, we **retain deleted chats** (and old branches of extant chats) by default. To delete local copies, manually delete the corresponding `.json` and any attached files from `backup_dir` or start from scratch by deleting `backup_dir`.

For Chrome et al., go to https://claude.ai in your browser, open **Developer tools** with F12 or Ctrl+Shift+I, navigate to the **Application** tab (it may be hidden under **⋮** > **More tools** > **Application**), and copy the value of the `sessionKey` cookie. (Firefox should be [similar](https://firefox-source-docs.mozilla.org/devtools-user/storage_inspector/index.html).)

    $ wl-paste | claude-backup

Your platform's equivalent of `wl-paste` might vary; I'm sure Claude knows what you should use ;) As this tool demonstrates **anyone with this cookie is authenticated as you on `claude.ai`** so be careful and never give this to anyone or anything you do not trust!

`claude.ai` is behind Cloudflare, and `claude-backup` can be unreliable behind a VPN. I swore to myself I was off the scrape-things-that-don't-want-to-be-scraped grind, so I recommend running `claude-backup` from a non-sus IP, bumping `--success-delay`, and lowering `--connections`. Any further workarounds are left as an exercise for the reader...

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

## Usage

    $ claude-backup --help
    usage: claude-backup [-h] [-v] [-c CONNECTIONS] [-d DELAY] [-r RETRIES]
                         [--min-retry-delay DELAY] [--max-retry-delay DELAY]
                         [backup_dir]

    Back up Claude.ai chats

    positional arguments:
      backup_dir            Directory to save backups (default:
                            ~/.local/share/claude-backup)

    options:
      -h, --help            show this help message and exit
      -v, --version         show program's version number and exit
      -c, --connections CONNECTIONS
                            Maximum concurrent connections (default: 6)
      -d, --success-delay DELAY
                            Delay after successful request in seconds (default:
                            0.25)
      -r, --retries RETRIES
                            Number of retries for API requests (default: 10)
      --min-retry-delay DELAY
                            Minimum retry delay in seconds (default: 1.0)
      --max-retry-delay DELAY
                            Maximum retry delay in seconds (default: 60.0)

## License

`claude-backup` is dedicated to the public domain where possible via CC0-1.0.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
