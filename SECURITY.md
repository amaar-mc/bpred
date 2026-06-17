# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

`bpred` is a pure-Python educational simulator with zero runtime dependencies
and no network access, secrets handling, or persistent storage. The attack
surface is minimal.

If you discover a security vulnerability (for example, a path traversal issue
in the CLI trace-file parser), please report it by opening a GitHub issue
marked with the `security` label. For sensitive matters that should not be
public, contact the maintainer directly via the email address listed in
`pyproject.toml`.

We aim to acknowledge reports within 72 hours and to release a fix within
14 days for confirmed vulnerabilities.
