#!/usr/bin/env python3
"""Statusline: branch, model, and context used.

Context comes from the transcript rather than the statusline payload, because
the payload's fields are not guaranteed across versions while the transcript's
`usage` block is what the API actually returned. Context used is
`input_tokens + cache_creation_input_tokens + cache_read_input_tokens` from the
most recent assistant message.

Every field degrades independently: a missing model, branch, or transcript drops
that segment instead of breaking the line. Run with --debug to dump the raw
payload and see what this version of Claude Code actually provides.
"""
import json
import os
import subprocess
import sys

# Real window sizes; the ladder picks the smallest one that still fits.
LIMITS = (200_000, 1_000_000)
WARN_AT = 0.75
ALERT_AT = 0.90

RESET, DIM, GREEN, YELLOW, RED = (
    "\033[0m", "\033[2m", "\033[32m", "\033[33m", "\033[31m")


def context_used(transcript_path):
    """Total prompt tokens on the last assistant message, or None."""
    if not transcript_path or not os.path.isfile(transcript_path):
        return None
    last = None
    try:
        with open(transcript_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                usage = (entry.get("message") or {}).get("usage")
                if isinstance(usage, dict):
                    last = usage
    except OSError:
        return None
    if not last:
        return None
    return sum(int(last.get(k, 0) or 0) for k in (
        "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))


def pick_limit(used):
    override = os.environ.get("CLAUDE_CTX_LIMIT")
    if override and override.isdigit():
        return int(override)
    for limit in LIMITS:
        if used <= limit:
            return limit
    return LIMITS[-1]


def fmt_context(used):
    limit = pick_limit(used)
    frac = used / limit if limit else 0.0
    color = RED if frac >= ALERT_AT else YELLOW if frac >= WARN_AT else GREEN
    short = f"{used / 1000:.0f}k"
    return f"{color}ctx {short}/{limit // 1000}k {frac * 100:.0f}%{RESET}"


def git_branch(cwd):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd or None, capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return None
    branch = out.stdout.strip()
    return branch if out.returncode == 0 and branch else None


def main():
    raw = sys.stdin.read()
    if "--debug" in sys.argv:
        print(raw)
        return
    try:
        payload = json.loads(raw)
    except ValueError:
        payload = {}

    cwd = (payload.get("workspace") or {}).get("current_dir") or payload.get("cwd")
    parts = []

    branch = git_branch(cwd)
    if branch:
        parts.append(f"\033[36m{branch}{RESET}")

    model = (payload.get("model") or {}).get("display_name")
    if model:
        parts.append(f"{DIM}{model}{RESET}")

    used = context_used(payload.get("transcript_path"))
    if used:
        parts.append(fmt_context(used))

    print(f"{DIM} | {RESET}".join(parts) if parts else "")


if __name__ == "__main__":
    main()
