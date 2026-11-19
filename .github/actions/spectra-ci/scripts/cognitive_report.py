#!/usr/bin/env python3
"""Cognitive complexity report of tracked Python files (SPCTR-305). Standard library only.

  cognitive_report.py --files LIST --base CHECKOUT  > cognitive.json

Parses each listed file with ``ast`` (repository code is never imported or
run) and prints {"tool": "cognitive", "functions": [...], "unparsed_files":
[...]}. A file that does not parse is listed, not fatal: it is the code under
review, not a failure of the tool. ``cognitive_complexity.py`` next to this
script is the single source of the algorithm; Spectra loads the same file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from cognitive_complexity import file_functions


def report(files: list[str], base: str) -> dict:
    functions, unparsed = [], []
    for relative in files:
        try:
            with open(os.path.join(base, relative), encoding='utf-8', errors='replace') as handle:
                source = handle.read()
            functions.extend(
                {'file': relative, 'function': item.name, 'start': item.start, 'end': item.end,
                 'cognitive': item.cognitive}
                for item in file_functions(source)
            )
        except (SyntaxError, ValueError, RecursionError, OSError):
            unparsed.append(relative)
    return {'tool': 'cognitive', 'functions': functions, 'unparsed_files': unparsed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--files', required=True, help='file listing the tracked Python paths, one per line')
    parser.add_argument('--base', required=True, help='repository checkout the paths are relative to')
    args = parser.parse_args(argv)
    with open(args.files, encoding='utf-8') as handle:
        files = [line.strip() for line in handle if line.strip()]
    json.dump(report(files, args.base), sys.stdout)
    return 0


if __name__ == '__main__':
    sys.exit(main())
