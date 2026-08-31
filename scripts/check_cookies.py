"""Validate an exported cookies.txt and prove it actually unblocks fetching.

Reports structure, domains and expiry, then does one real transcript fetch.
Deliberately never prints a cookie value — only names, domains and counts —
so the output is safe to paste into a bug report.

    python scripts/check_cookies.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config  # noqa: E402

# Cookies that actually carry the signed-in session. Missing these means the
# export captured a logged-out browser, which looks valid but changes nothing.
SESSION_COOKIES = {"SID", "SSID", "HSID", "SAPISID", "APISID", "__Secure-1PSID"}


def main() -> int:
    path_str = config.YTDLP_COOKIES_FILE
    if not path_str:
        print("No cookie file configured.")
        print("  Export one to secrets/cookies.txt (see secrets/README.md),")
        print("  or set YTDLP_COOKIES_FILE=/path/to/cookies.txt in .env")
        return 1

    path = Path(path_str)
    if not path.exists():
        print(f"Configured cookie file does not exist: {path}")
        return 1

    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    entries, malformed = [], 0
    for line in raw:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            malformed += 1
            continue
        entries.append(parts)

    if not entries:
        print(f"{path}: no cookie entries found.")
        print("  The file must be Netscape format (tab-separated), not JSON.")
        return 1

    domains: dict[str, int] = {}
    names: set[str] = set()
    now = int(time.time())
    expired = 0
    soonest = None
    for parts in entries:
        domain, _, _, _, expiry, name = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
        domains[domain] = domains.get(domain, 0) + 1
        names.add(name)
        try:
            exp = int(expiry)
        except ValueError:
            continue
        if exp == 0:
            continue  # session cookie
        if exp < now:
            expired += 1
        elif soonest is None or exp < soonest:
            soonest = exp

    yt = {d: n for d, n in domains.items() if "youtube" in d or "google" in d}
    print(f"file:        {path}")
    print(f"entries:     {len(entries)}" + (f" ({malformed} malformed lines skipped)" if malformed else ""))
    print(f"domains:     {len(domains)} total, {len(yt)} youtube/google")
    for d, n in sorted(yt.items(), key=lambda kv: -kv[1])[:6]:
        print(f"               {d:<28}{n}")

    present = SESSION_COOKIES & names
    missing = SESSION_COOKIES - names
    print(f"session:     {len(present)}/{len(SESSION_COOKIES)} session cookies present")
    if missing:
        print(f"               missing: {', '.join(sorted(missing))}")

    if expired:
        print(f"expiry:      {expired} entr(ies) already expired")
    if soonest:
        days = (soonest - now) / 86400
        print(f"             earliest live expiry in {days:.1f} days")

    problems = []
    if not yt:
        problems.append("no youtube.com/google.com cookies — export was for the wrong site")
    if not present:
        problems.append("no session cookies — the browser was signed out when exported")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\nStructure looks good. Testing a real fetch…")
    from backend.connectors import check_youtube_access

    result = check_youtube_access()
    if result["ok"]:
        print(f"SUCCESS — transcripts reachable via {result['method']} ({result['detail']})")
        print("Run the sweep: the backlog of approved videos will ingest.")
        return 0
    print(f"STILL FAILING — {result['hint']}")
    print(f"  detail: {result['detail'][:300]}")
    if result.get("blocked"):
        print("\n  The cookies loaded but YouTube still refuses. Usually means:")
        print("   - the export came from a signed-out profile, or")
        print("   - this IP is in a hard cool-off that authentication alone won't lift.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
