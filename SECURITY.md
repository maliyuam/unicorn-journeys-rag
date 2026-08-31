# Security

## Reporting a vulnerability

Please report security issues privately through GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
(**Security → Report a vulnerability** on this repository) rather than opening a
public issue. Include what you did, what happened, and what you expected.

## What this project handles

Running it means handling three kinds of credential. None of them belong in
git, and all three are ignored by [.gitignore](.gitignore):

| Credential | Lives in | If it leaks |
|---|---|---|
| `ANTHROPIC_API_KEY` | `.env` | Revoke it in the Anthropic Console and issue a new one. It is billable. |
| `MONGODB_URI` (contains the database password) | `.env` | Rotate the password in Atlas → Database Access. The URI grants read/write to your corpus. |
| YouTube session cookies | `secrets/cookies.txt` | Sign out of Google in that browser profile, which invalidates the exported session. |

Two habits matter more than any tooling here:

- **`.env` is the only place credentials go.** Nothing in `backend/` reads a
  hardcoded secret, and `.env.example` carries placeholders only. If you add a
  new credential, add it to `.env.example` as a commented placeholder and never
  with a real value.
- **Cookies are a password.** `secrets/cookies.txt` is a live signed-in Google
  session, not configuration. `scripts/check_cookies.py` validates it without
  ever printing a cookie value, so its output is safe to paste into an issue.

## What is scanned

Every push and pull request runs [gitleaks](https://github.com/gitleaks/gitleaks)
over the **full history**, not just the diff, so a secret that was committed and
later deleted is still caught. If it fires on your branch, rotate the credential
first and rewrite the history second — a secret that reached a public remote
should be treated as compromised even after a force-push.

## Deployment notes

The app ships **no authentication**. Every endpoint — including
`POST /api/clear`, which drops the index, and `DELETE /api/sources/{id}` — is
open to anyone who can reach the port. That is fine for `localhost` and wrong
for the public internet. If you deploy it beyond your own machine, put it behind
an authenticating reverse proxy or your platform's access control.

The Docker image runs as a non-root user (uid 10001) and takes credentials at
run time via `--env-file`, so they are never baked into an image layer.
