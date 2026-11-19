#!/usr/bin/env python3
"""Runner helper for quality_provider=linters (SPCTR-287). Standard library only.

  changed-lines            unified diff (git diff -U0) on stdin -> {path: [[start, end]]}
  new-code ...             the new code reference block sent with every payload (SPCTR-303)
  payload KIND REPORT ...  JSON body for /results/{sarif,coverage,metrics,manifest}
  count-results SARIF      number of results in a SARIF log; exit 1 if unreadable
  manifest-entry ...       one JSON line of the execution manifest (SPCTR-301)

Spectra CI never uploads source code: SARIF snippets / fixes and jscpd code
fragments are stripped here, before anything leaves the runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from typing import Any
from urllib.parse import unquote

_HUNK = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@')
_SOURCE_KEYS = frozenset({'snippet', 'contextRegion', 'fixes'})


def _diff_path(target: str) -> str | None:
    """Path of a ``+++`` header; C-quoted paths (core.quotePath) are decoded."""
    target = target.rstrip('\t').strip()
    if target == '/dev/null':
        return None
    if len(target) >= 2 and target.startswith('"') and target.endswith('"'):
        target = _unquote_c(target[1:-1])
    return target.removeprefix('b/')


_C_ESCAPES = {'a': 7, 'b': 8, 't': 9, 'n': 10, 'v': 11, 'f': 12, 'r': 13, '"': 34, '\\': 92}


def _unquote_c(text: str) -> str:
    """Decode git's C-style quoting: octal UTF-8 bytes and backslash escapes."""
    data = bytearray()
    index = 0
    while index < len(text):
        char = text[index]
        if char != '\\' or index + 1 == len(text):
            data += char.encode('utf-8')
            index += 1
            continue
        following = text[index + 1]
        octal = text[index + 1:index + 4]
        if len(octal) == 3 and all(c in '01234567' for c in octal):
            data.append(int(octal, 8) & 0xFF)
            index += 4
        elif following in _C_ESCAPES:
            data.append(_C_ESCAPES[following])
            index += 2
        else:
            data += following.encode('utf-8')
            index += 2
    return data.decode('utf-8', errors='replace')


def parse_changed_lines(diff_text: str) -> dict[str, list[list[int]]]:
    """Added or modified line ranges of the new file side, per path."""
    changed: dict[str, list[list[int]]] = {}
    path = None
    previous = ''
    for line in diff_text.splitlines():
        # A '+++' line is a file header only right after '---': an added line
        # whose content starts with '++ ' looks the same inside a hunk.
        is_header = line.startswith('+++ ') and previous.startswith('--- ')
        previous = line
        if is_header:
            path = _diff_path(line[4:])
            continue
        match = _HUNK.match(line)
        if not match or path is None:
            continue
        start = int(match.group(1))
        count = 1 if match.group(2) is None else int(match.group(2))
        if count > 0:
            changed.setdefault(path, []).append([start, start + count - 1])
    return changed


NEW_CODE_METHODS = frozenset({
    'explicit', 'merge-base', 'push-before', 'default-branch-merge-base', 'parent', 'none',
})


def new_code_block(reference: str, method: str, reason: str) -> dict[str, Any]:
    """``{"status": "ok", "reference": sha, "method": ...}`` or ``{"status": "unknown", "reason": ...}``."""
    reference = (reference or '').strip().lower()
    if re.fullmatch(r'[0-9a-f]{40}', reference) and method in NEW_CODE_METHODS - {'none'}:
        return {'status': 'ok', 'reference': reference, 'method': method}
    return {'status': 'unknown', 'reference': None, 'method': 'none', 'reason': (reason or 'no reference')[:500]}


LINE_HASH_KEY = 'spectra/lineHash/v1'


def line_content_hash(line: str) -> str:
    """Same value as app.fingerprint.line_content_hash (SPCTR-302): whitespace-free sha256."""
    return hashlib.sha256(''.join(line.split()).encode('utf-8')).hexdigest()[:32]


def _source_path(uri: str, base_path: str) -> str | None:
    path = unquote(uri)
    if path.lower().startswith('file://'):
        path = path[len('file://'):]
    if not os.path.isabs(path):
        path = os.path.join(base_path or '.', path)
    real = os.path.realpath(path)
    root = os.path.realpath(base_path or '.')
    # Only files inside the checkout are read: a SARIF log cannot point the
    # runner at arbitrary files.
    return real if real.startswith(root + os.sep) and os.path.isfile(real) else None


def _result_line(result: Any) -> tuple[str, int] | None:
    try:
        physical = result['locations'][0]['physicalLocation']
        return physical['artifactLocation']['uri'], int(physical['region']['startLine'])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def add_line_hashes(document: Any, base_path: str) -> Any:
    """Add the hash of each result's source line; the line itself never leaves the runner."""
    cache: dict[str, list[str] | None] = {}
    for run in (document.get('runs') or []) if isinstance(document, dict) else []:
        for result in run.get('results') or []:
            located = _result_line(result)
            if not located:
                continue
            uri, line_number = located
            if uri not in cache:
                path = _source_path(uri, base_path)
                try:
                    with open(path, encoding='utf-8', errors='replace') as handle:  # type: ignore[arg-type]
                        cache[uri] = handle.read().splitlines()
                except (OSError, TypeError):
                    cache[uri] = None
            lines = cache[uri]
            if lines and 0 < line_number <= len(lines):
                fingerprints = result.setdefault('partialFingerprints', {})
                fingerprints[LINE_HASH_KEY] = line_content_hash(lines[line_number - 1])
    return document


def strip_source(node: Any) -> Any:
    """Drop SARIF keys that may carry source code (snippets, context, fixes)."""
    if isinstance(node, dict):
        return {key: strip_source(value) for key, value in node.items() if key not in _SOURCE_KEYS}
    if isinstance(node, list):
        return [strip_source(item) for item in node]
    return node


def strip_jscpd_fragments(report: dict[str, Any]) -> dict[str, Any]:
    """jscpd duplicates carry the duplicated code in ``fragment``; keep locations only."""
    duplicates = [
        {key: value for key, value in duplicate.items() if key != 'fragment'}
        for duplicate in report.get('duplicates') or []
        if isinstance(duplicate, dict)
    ]
    return {**report, 'duplicates': duplicates}


def _read_json(path: str) -> Any:
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def _read_text(path: str) -> str:
    with open(path, encoding='utf-8') as handle:
        return handle.read()


MANIFEST_STATUSES = ('ok', 'failed', 'unavailable')
MAX_REASON_LENGTH = 500


def count_results(report_path: str) -> int:
    """Results of a SARIF 2.1 log; raises ValueError when it is not one."""
    document = _read_json(report_path)
    runs = document.get('runs') if isinstance(document, dict) else None
    if not isinstance(runs, list):
        raise ValueError(f'{report_path} is not a SARIF log (no runs array)')
    return sum(len(run.get('results') or []) for run in runs if isinstance(run, dict))


def manifest_entry(tool: str, status: str, version: str, results: str, reason: str, seconds: str) -> dict[str, Any]:
    """One tool's execution record; numeric fields stay null when unknown."""
    if status not in MANIFEST_STATUSES:
        raise ValueError(f'unknown tool status: {status}')
    return {
        'tool': tool,
        'status': status,
        'version': version or None,
        'results': int(results) if results.strip().isdigit() else None,
        'reason': reason.strip()[:MAX_REASON_LENGTH] or None,
        'seconds': int(seconds) if seconds.strip().isdigit() else None,
    }


def _read_manifest(path: str) -> list[dict[str, Any]]:
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_lines(path: str | None) -> list[str]:
    if not path:
        return []
    with open(path, encoding='utf-8') as handle:
        return [line.strip() for line in handle if line.strip()]


def build_payload(
    kind: str, tool: str, report_path: str, base_path: str, changed_path: str | None,
    excludes_path: str | None = None, new_code_path: str | None = None,
) -> dict[str, Any]:
    new_code = _read_json(new_code_path) if new_code_path else None
    if kind == 'manifest':
        manifest = {'tools': _read_manifest(report_path), 'default_excludes': _read_lines(excludes_path)}
        if new_code is not None:
            manifest['new_code'] = new_code
        return manifest
    payload: dict[str, Any] = {'base_path': base_path}
    if new_code is not None:
        payload['new_code'] = new_code
    if changed_path:
        payload['changed_lines'] = _read_json(changed_path)
    if kind == 'sarif':
        payload.update(tool=tool, sarif=add_line_hashes(strip_source(_read_json(report_path)), base_path))
    elif kind == 'coverage':
        payload.update(format='cobertura', report=_read_text(report_path))
    elif tool == 'jscpd':
        payload.update(tool=tool, report=strip_jscpd_fragments(_read_json(report_path)))
    else:
        payload.update(tool=tool, report=_read_text(report_path))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('changed-lines')
    payload = sub.add_parser('payload')
    payload.add_argument('kind', choices=('sarif', 'coverage', 'metrics', 'manifest'))
    payload.add_argument('report')
    payload.add_argument('--tool', default='')
    payload.add_argument('--base-path', default='')
    payload.add_argument('--changed-lines', default=None)
    payload.add_argument('--excludes', default=None)
    payload.add_argument('--new-code', default=None)
    new_code = sub.add_parser('new-code')
    for option in ('--reference', '--method', '--reason'):
        new_code.add_argument(option, default='')
    count = sub.add_parser('count-results')
    count.add_argument('report')
    entry = sub.add_parser('manifest-entry')
    for option in ('--tool', '--status', '--version', '--results', '--reason', '--seconds'):
        entry.add_argument(option, default='')
    args = parser.parse_args(argv)

    if args.command == 'changed-lines':
        json.dump(parse_changed_lines(sys.stdin.read()), sys.stdout)
    elif args.command == 'new-code':
        json.dump(new_code_block(args.reference, args.method, args.reason), sys.stdout)
    elif args.command == 'count-results':
        try:
            print(count_results(args.report))
        except (OSError, ValueError) as exc:
            print(f'unreadable SARIF: {exc}', file=sys.stderr)
            return 1
    elif args.command == 'manifest-entry':
        entry_fields = (args.tool, args.status, args.version, args.results, args.reason, args.seconds)
        print(json.dumps(manifest_entry(*entry_fields)))
    else:
        json.dump(build_payload(
            args.kind, args.tool, args.report, args.base_path, args.changed_lines, args.excludes, args.new_code,
        ), sys.stdout)
    return 0


if __name__ == '__main__':
    sys.exit(main())
