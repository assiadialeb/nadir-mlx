#!/usr/bin/env python3
"""Convert linter native reports to SARIF 2.1.0 (SPCTR-289). Standard library only.

Formats verified against the pinned releases (see docs/dev/LINTERS.md):
  shellcheck  json1   {"comments": [{file, line, endLine, column, endColumn, level, code, message}]}
  tflint      json    {"issues": [{rule: {name, severity}, message, range: {filename, start, end}}]}
  yamllint    parsable text lines "path:line:col: [severity] message (rule)"
  sqlfluff    json    [{"filepath": ..., "violations": [{start_line_no, start_line_pos, code, description}]}]
  buf         jsonl   {"path", "start_line", "type", "message"} per line

Levels come from linters/levels.json (SPCTR-304): one ordered native ->
SARIF table per tool. A rule listed there as ``unparsed`` (sqlfluff PRS,
buf COMPILE, yamllint syntax) means the file could not be analysed: it is
reported as a SARIF tool execution notification, never as a finding.

The runner never uploads source code: SARIF output carries locations and
messages only, and Spectra strips snippets again on ingest.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any
from urllib.parse import quote


_YAML_LINE = re.compile(r'^(?P<path>.+?):(?P<line>\d+):(?P<col>\d+): \[(?P<level>[^\]]+)\] (?P<msg>.*) \((?P<rule>[^)]+)\)$')
LEVELS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'linters', 'levels.json')


def load_levels(path: str = LEVELS_FILE) -> dict[str, dict[str, Any]]:
    with open(path, encoding='utf-8') as handle:
        table = json.load(handle)
    return {tool: spec for tool, spec in table.items() if not tool.startswith('$')}


_LEVELS = load_levels()


def _uri(path: str) -> str:
    """Percent-encoded relative URI; Spectra decodes it on ingest, so a path with
    a space or an accent keeps its exact name (replacing them broke matching)."""
    return quote(path.replace('\\', '/'), safe='/._-~')


def _level(tool: str, raw: str) -> str:
    """SARIF level of a native level; an unknown native level reads as warning."""
    return _LEVELS[tool]['levels'].get((raw or '').strip().lower(), 'warning')


def _is_unparsed(tool: str, rule_id: str) -> bool:
    return rule_id in _LEVELS[tool].get('unparsed', ())


def _unparsed(rule_id: str, message: str, path: str, line: int) -> dict[str, Any]:
    """A file the tool could not analyse (SARIF toolExecutionNotification)."""
    return {
        'level': 'error',
        'descriptor': {'id': rule_id},
        'message': {'text': message[:500]},
        'locations': [{'physicalLocation': {
            'artifactLocation': {'uri': _uri(path)}, 'region': {'startLine': max(1, line)},
        }}],
        'properties': {'spectra/unparsed': True},
    }


def _result(rule_id: str, level: str, message: str, path: str, line: int, column: int = 1) -> dict[str, Any]:
    return {
        'ruleId': rule_id,
        'level': level,
        'message': {'text': message},
        'locations': [{
            'physicalLocation': {
                'artifactLocation': {'uri': _uri(path)},
                'region': {'startLine': max(1, line), 'startColumn': max(1, column)},
            },
        }],
    }


def _rules(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = sorted({result['ruleId'] for result in results})
    return [{'id': rule_id} for rule_id in ids]


def _document(
    results: list[dict[str, Any]], tool: str, version: str, unparsed: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    run: dict[str, Any] = {
        'tool': {'driver': {'name': tool, 'version': version, 'rules': _rules(results)}},
        'results': results,
    }
    if unparsed:
        run['invocations'] = [{'executionSuccessful': True, 'toolExecutionNotifications': unparsed}]
    return {
        '$schema': 'https://json.schemastore.org/sarif-2.1.0.json',
        'version': '2.1.0',
        'runs': [run],
    }


def _split(tool: str, entries: list[tuple[str, str, str, str, int, int]], version: str) -> dict[str, Any]:
    """(rule, native level, message, path, line, column) -> SARIF with unparsed files apart."""
    results, unparsed = [], []
    for rule_id, native, message, path, line, column in entries:
        if _is_unparsed(tool, rule_id):
            unparsed.append(_unparsed(rule_id, message, path, line))
        else:
            results.append(_result(rule_id, _level(tool, native), message, path, line, column))
    return _document(results, tool, version, unparsed)


def convert_shellcheck(report_path: str, version: str) -> dict[str, Any]:
    with open(report_path, encoding='utf-8') as handle:
        comments = json.load(handle).get('comments') or []
    return _split('shellcheck', [(
        f"SC{comment.get('code', 0)}",
        str(comment.get('level', 'warning')),
        str(comment.get('message', '')),
        comment.get('file', ''),
        int(comment.get('line', 1) or 1),
        int(comment.get('column', 1) or 1),
    ) for comment in comments], version)


def convert_tflint(report_path: str, version: str) -> dict[str, Any]:
    """tflint reports its own failures (bad option, config, plugin) in "errors"."""
    with open(report_path, encoding='utf-8') as handle:
        report = json.load(handle)
    errors = [str(error.get('message', error)) for error in report.get('errors') or [] if error]
    if errors:
        raise SystemExit(f"tflint reported errors: {'; '.join(errors)[:400]}")
    return _split('tflint', [(
        issue.get('rule', {}).get('name', 'tflint'),
        str(issue.get('rule', {}).get('severity', 'warning')),
        str(issue.get('message', '')),
        issue.get('range', {}).get('filename', ''),
        int(issue.get('range', {}).get('start', {}).get('line', 1) or 1),
        int(issue.get('range', {}).get('start', {}).get('column', 1) or 1),
    ) for issue in report.get('issues') or []], version)


def convert_yamllint(report_path: str, version: str) -> dict[str, Any]:
    entries = []
    with open(report_path, encoding='utf-8') as handle:
        for line in handle:
            match = _YAML_LINE.match(line.rstrip('\n'))
            if match:
                entries.append((
                    match.group('rule'), match.group('level'), match.group('msg'),
                    match.group('path'), int(match.group('line')), int(match.group('col')),
                ))
    return _split('yamllint', entries, version)


def convert_sqlfluff(report_path: str, version: str) -> dict[str, Any]:
    """A rule violation is a warning; one sqlfluff flags as a warning is a note."""
    with open(report_path, encoding='utf-8') as handle:
        files = json.load(handle)
    entries = []
    for entry in files if isinstance(files, list) else []:
        for violation in entry.get('violations') or []:
            entries.append((
                str(violation.get('code', 'sqlfluff')),
                'warning' if violation.get('warning') else 'violation',
                str(violation.get('description', '')),
                entry.get('filepath', ''),
                int(violation.get('start_line_no', 1) or 1),
                int(violation.get('start_line_pos', 1) or 1),
            ))
    return _split('sqlfluff', entries, version)


def convert_buf(report_path: str, version: str) -> dict[str, Any]:
    """buf lint events carry no level: a lint rule is a warning, COMPILE an unparsed file."""
    entries = []
    with open(report_path, encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            event = json.loads(line)
            entries.append((
                str(event.get('type', 'buf')),
                'lint',
                str(event.get('message', '')),
                event.get('path', ''),
                int(event.get('start_line', 1) or 1),
                int(event.get('start_column', 1) or 1),
            ))
    return _split('buf', entries, version)


_CONVERTERS = {
    'shellcheck': convert_shellcheck,
    'tflint': convert_tflint,
    'yamllint': convert_yamllint,
    'sqlfluff': convert_sqlfluff,
    'buf': convert_buf,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('tool', choices=sorted(_CONVERTERS))
    parser.add_argument('report')
    parser.add_argument('output')
    parser.add_argument('--version', default='0')
    args = parser.parse_args(argv)
    document = _CONVERTERS[args.tool](args.report, args.version)
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(document, handle)
    return 0


if __name__ == '__main__':
    sys.exit(main())
