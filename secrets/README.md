# secrets/

Everything in this directory is git-ignored. Nothing here should ever be
committed, pasted into a chat, or shared.

## cookies.txt — authenticating YouTube transcript fetches

When YouTube rate-limits this network ("Sign in to confirm you're not a bot"),
authenticating with your own browser session restores access. yt-dlp reads
cookies in the standard Netscape format.

**Export it (one time, ~2 minutes):**

1. Install a cookie-export extension that works locally, for example
   "Get cookies.txt LOCALLY" (Chrome/Edge) or "cookies.txt" (Firefox).
   Prefer an open-source one that exports on-device — never a service that
   uploads your cookies.
2. Open <https://www.youtube.com> and make sure you are signed in.
3. Use the extension to export cookies **for youtube.com** in
   Netscape/`cookies.txt` format.
4. Save the file here as `secrets/cookies.txt`.

The app picks it up automatically — no `.env` edit needed. To keep it
elsewhere, set `YTDLP_COOKIES_FILE=/path/to/cookies.txt` instead.

**Verify it:**

```bash
python scripts/check_cookies.py
```

That checks the format and expiry and then does one real fetch, without ever
printing a cookie value.

## Handle with care

- These cookies grant access to your signed-in Google session. Treat the file
  like a password.
- They expire; re-export when `check_cookies.py` reports them stale.
- Revoke anytime by signing out of Google in that browser profile, which
  invalidates the exported session.
- Use an account you are comfortable automating with. A secondary Google
  account is a reasonable precaution.
