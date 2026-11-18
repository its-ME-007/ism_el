"""Parse a bounded, cutoff-usable sample of the public Enron email corpus.

The Enron dataset (released by FERC, public domain) ships as a maildir tree,
distributed as a ~400MB tarball at https://www.cs.cmu.edu/~enron/ . This reads
from any of:

  * a ``.tar.gz`` tarball (streamed, not fully extracted),
  * an already-extracted ``maildir`` directory,
  * a URL (downloaded once to a local cache, then streamed).

It emits one JSONL record per email with the fields the adapter needs:
participant addresses (the real access-group signal), the ``Date`` header,
subject, and plain-text body. Everything is stdlib only; no new dependencies.

Example::

    python -m src.tiering.ingestion.fetch_enron \
        --source data/raw/enron_mail.tar.gz \
        --output data/processed/enron_sample.jsonl --max-emails 2000
"""

from __future__ import annotations

import argparse
from datetime import timezone
import email
from email.message import Message
from email.utils import getaddresses, parsedate_to_datetime
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen


DEFAULT_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"


def _address_set(message: Message, headers=("from", "to", "cc")) -> list[str]:
    """Normalized lowercase addresses across the given headers, order-stable."""
    raw = []
    for header in headers:
        raw.extend(message.get_all(header, []))
    seen: dict[str, None] = {}
    for _name, addr in getaddresses(raw):
        addr = addr.strip().lower()
        if addr and "@" in addr:
            seen.setdefault(addr, None)
    return list(seen)


def _plain_body(message: Message) -> str:
    """Decoded text/plain body (Enron messages are plain text)."""
    if message.is_multipart():
        parts = [part for part in message.walk()
                 if part.get_content_type() == "text/plain"]
    else:
        parts = [message]
    chunks = []
    for part in parts:
        payload = part.get_payload(decode=True)
        if payload is None:
            continue
        charset = part.get_content_charset() or "latin-1"
        chunks.append(payload.decode(charset, errors="replace"))
    return "\n".join(chunks).strip()


def _record(raw_bytes: bytes, path: str) -> dict | None:
    """Parse one raw message into a JSON-able record, or None if unusable."""
    message = email.message_from_bytes(raw_bytes)
    try:
        when = parsedate_to_datetime(message.get("date", ""))
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    body = _plain_body(message)
    if not body:
        return None
    sender = _address_set(message, ("from",))
    recipients = _address_set(message, ("to", "cc"))
    participants = list(dict.fromkeys(sender + recipients))
    if not participants:
        return None
    # Owning mailbox user = top maildir folder (e.g. "allen-p/inbox/3." -> allen-p).
    owner = _owner(path)
    return {
        "message_id": message.get("message-id", path).strip() or path,
        "path": path,
        "date": when.astimezone(timezone.utc).isoformat(),
        "subject": (message.get("subject", "") or "").strip(),
        "body": body,
        "sender": sender,
        "recipients": recipients,
        "participants": participants,
        "owner": owner,
        "source": "CMU Enron email corpus (FERC public domain)",
    }


def _owner(path: str) -> str:
    return path.replace("\\", "/").split("maildir/")[-1].split("/")[0]


def _iter_messages(source: Path, owners: set[str] | None = None):
    """Yield (path, raw_bytes) for each email file under a tarball or directory.

    ``owners`` keeps whole mailboxes (e.g. ``{"kaminski-v"}``) instead of the
    first N files, which in a tarball are alphabetical by mailbox.
    """
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file() and (not owners or _owner(str(path)) in owners):
                yield str(path), path.read_bytes()
        return
    with tarfile.open(source, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if owners and _owner(member.name) not in owners:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            yield member.name, handle.read()


def _ensure_local(source: str, cache: Path) -> Path:
    """Return a local path, downloading a URL to the cache once if needed."""
    if "://" not in source:
        return Path(source)
    if cache.exists():
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {source} -> {cache} (large, one-time)")
    with urlopen(source) as response, cache.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    return cache


def fetch(source: str, output: Path, max_emails: int = 0,
          cache: Path = Path("data/raw/enron_mail.tar.gz"),
          owners: set[str] | None = None) -> int:
    """Write up to ``max_emails`` (0 = all) parsed records to ``output`` JSONL."""
    local = _ensure_local(source, cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8", newline="\n") as destination:
        for path, raw in _iter_messages(local, owners):
            record = _record(raw, path)
            if record is None:
                continue
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written % 500 == 0:
                print(f"Parsed {written} emails")
            if max_emails and written >= max_emails:
                break
    print(f"Wrote {written} email records to {output}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=DEFAULT_URL,
                        help="Tarball path, maildir directory, or URL "
                             f"(default: {DEFAULT_URL})")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-emails", type=int, default=0,
                        help="Maximum emails to parse; 0 means all")
    parser.add_argument("--cache", type=Path, default=Path("data/raw/enron_mail.tar.gz"),
                        help="Where a downloaded URL tarball is cached")
    parser.add_argument("--owners", default="",
                        help="Comma-separated mailbox names to keep (e.g. kaminski-v,lay-k)")
    args = parser.parse_args()
    owners = {name.strip() for name in args.owners.split(",") if name.strip()} or None
    fetch(args.source, args.output, args.max_emails, args.cache, owners)


if __name__ == "__main__":
    main()
