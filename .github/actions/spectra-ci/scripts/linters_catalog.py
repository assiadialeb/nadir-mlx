#!/usr/bin/env python3
"""Catalog queries for the linters phase (SPCTR-289). Standard library only.

  paths <tool>   print the tool's git pathspecs, one per line
  asset <tool>   print the allowlist asset name for the tool's Linux artifact
  tools          print the core tool names, one per line
  excludes       print the default exclusion globs, one per line (SPCTR-301)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('catalog')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('tools')
    sub.add_parser('excludes')
    paths = sub.add_parser('paths')
    paths.add_argument('tool')
    asset = sub.add_parser('asset')
    asset.add_argument('tool')
    args = parser.parse_args(argv)

    catalog = load(args.catalog)
    if args.command == 'tools':
        for tool in catalog.get('core_tools') or {}:
            print(tool)
    elif args.command == 'excludes':
        for pattern in catalog.get('default_excludes') or []:
            print(pattern)
    elif args.command == 'paths':
        for pathspec in (catalog.get('core_tools', {}).get(args.tool, {}).get('paths')) or []:
            print(pathspec)
    elif args.command == 'asset':
        allowlist = load(str(Path(args.catalog).parent.parent / 'tools' / 'allowlist.json'))
        assets = allowlist['tools'][args.tool]['assets']
        print(next(iter(assets)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
