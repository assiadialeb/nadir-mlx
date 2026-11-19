"""Emit the mirror sync manifest from the tool allowlist (SPCTR-291, SPCTR-289).

Each line is `tool|version|tag|asset|repo|sha256`, consumed by
.forgejo/workflows/sync-tool-mirror.yml to fetch and publish every pinned
artifact into the Forgejo generic package registry. `tag` follows the
allowlist's `tag_template` (default `v{version}`; biome tags releases as
`@biomejs/biome@x.y.z`).
"""

from __future__ import annotations

import json
import sys


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: mirror_manifest.py <allowlist.json>", file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as handle:
        allowlist = json.load(handle)
    for tool_name, tool in sorted((allowlist.get("tools") or {}).items()):
        upstream = tool.get("upstream") or {}
        repo = upstream.get("repo", "")
        tag = (upstream.get("tag_template") or "v{version}").replace(
            "{version}", tool["version"]
        )
        for asset_name, asset in sorted((tool.get("assets") or {}).items()):
            fields = (tool_name, tool["version"], tag, asset_name, repo, asset["sha256"])
            print("|".join(fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
