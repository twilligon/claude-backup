# `claude-backup`

Backup Claude.ai chats to local JSON files.

`claude-backup` fetches all your Claude.ai chat conversations and saves them as JSON files. It uses an incremental sync algorithm to efficiently update only changed or new conversations. The tool handles authentication via browser cookies or a session key, supports multiple organizations, and preserves chat metadata including renames.

## Features

`claude-backup` is the product of reverse-engineering Claude.ai's internal API (for Good, not Evil---please don't ban me Anthropic 🙏) so I can't make any guarantees this tool will continue to work. That said, it works as described as of 2025-10-26, and I'll likely update it best-effort when things break. Barring that, PRs welcome ;)

In this project we save the raw chat JSON from Claude.ai's API without parsing or normalizing it. This should be a bit more resilient than approaches that normalize it to some internal data model that could diverge from that of Claude.ai, but it means we don't e.g. download resources linked from within the chat. In practice, this preserves all text content and attachments, but not necessarily images, PDFs, container uploads, "advanced research" reports, etc.

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
    usage: claude-backup [-h] [-r RETRIES] [-s SUCCESS_DELAY]
                         [--min-retry-delay MIN_RETRY_DELAY]
                         [--max-retry-delay MAX_RETRY_DELAY] [--force-refresh]
                         [backup_dir]

    Backup Claude.ai chats

    positional arguments:
      backup_dir            Directory to save backups (default:
                            ~/.local/share/claude-backup)

    options:
      -h, --help            show this help message and exit
      -r, --retries RETRIES
                            Number of retries for API requests (default: 10)
      -s, --success-delay SUCCESS_DELAY
                            Delay after successful request in seconds (default:
                            0.25)
      --min-retry-delay MIN_RETRY_DELAY
                            Minimum retry delay in seconds (default: 1)
      --max-retry-delay MAX_RETRY_DELAY
                            Maximum retry delay in seconds (default: 60)
      --force-refresh       Re-fetch all accounts and chats, ignoring cache
                            (default: False)

## Authentication

Set the `CLAUDE_SESSION_KEY` environment variable to your claude.ai sessionKey cookie. If not set, the tool will attempt to load cookies from your browser automatically.

## License

`claude-backup` is dedicated to the public domain where possible via CC0-1.0.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
