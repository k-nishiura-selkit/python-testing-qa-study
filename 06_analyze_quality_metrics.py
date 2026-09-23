#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import os
import re
import time
from pathlib import Path, PurePosixPath
from typing import Iterable

import pandas as pd

REPO_METRICS = Path('data/processed/04_repository_metrics.csv')
TEST_ARTIFACTS = Path('data/processed/04_test_artifacts.csv')
SELECTED = Path('data/processed/selected_repositories.csv')
OUTPUT = Path('data/processed/04_quality_metrics.csv')
PILOT_OUTPUT = Path('data/processed/04_quality_metrics_pilot.csv')

COHORTS = ['pre_ai', 'contemporary']
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_PYTHON_BYTES = 2 * 1024 * 1024
SKIP_DIRS = {
    '.git', '.hg', '.svn', '.venv', 'venv', 'env', '.env',
    'site-packages', 'dist-packages', 'node_modules', '__pycache__',
    '.tox', '.nox', 'build', 'dist', '.eggs', 'eggs',
    '.mypy_cache', '.pytest_cache', '.ruff_cache',
}

CI_PREFIXES = (
    '.github/workflows/', '.gitlab-ci.yml', '.travis.yml',
    '.circleci/', 'azure-pipelines.yml',
)
PRECOMMIT_PATH = '.pre-commit-config.yaml'

CONFIG_NAMES = {
    'pyproject.toml', 'setup.cfg', 'tox.ini', '.flake8', '.pylintrc',
    'pylintrc', 'mypy.ini', '.mypy.ini', 'pyrightconfig.json',
}
DEPENDENCY_NAMES = {
    'requirements.txt', 'requirements-dev.txt', 'requirements_test.txt',
    'requirements-test.txt', 'dev-requirements.txt', 'pyproject.toml',
    'setup.cfg', 'setup.py', 'pipfile', 'pdm.lock', 'poetry.lock',
}
README_RE = re.compile(r'(?i)^readme(?:\.[^.]+)?$')

UNITTEST_ASSERTS = {
    'assertEqual', 'assertNotEqual', 'assertTrue', 'assertFalse',
    'assertIs', 'assertIsNot', 'assertIsNone', 'assertIsNotNone',
    'assertIn', 'assertNotIn', 'assertIsInstance', 'assertNotIsInstance',
    'assertGreater', 'assertGreaterEqual', 'assertLess', 'assertLessEqual',
    'assertRegex', 'assertNotRegex', 'assertCountEqual',
    'assertAlmostEqual', 'assertNotAlmostEqual', 'assertSequenceEqual',
    'assertListEqual', 'assertTupleEqual', 'assertSetEqual',
    'assertDictEqual', 'assertMultiLineEqual', 'assertRaises',
    'assertRaisesRegex', 'assertWarns', 'assertWarnsRegex',
    'assertLogs', 'assertNoLogs',
}
MOCK_ASSERTS = {
    'assert_called', 'assert_not_called', 'assert_called_once',
    'assert_called_with', 'assert_called_once_with', 'assert_any_call',
    'assert_has_calls', 'assert_awaited', 'assert_not_awaited',
    'assert_awaited_once', 'assert_awaited_with',
    'assert_awaited_once_with', 'assert_any_await', 'assert_has_awaits',
}

TOOL_FAMILIES = {
    'lint': {
        'ruff': {
            'config': [r'(?im)^\s*\[tool\.ruff(?:\.|\])'],
            'dep': r'(?im)^\s*ruff(?:\[[^\]]+\])?\s*(?:[=<>!~]|$)',
            'cmd': r'(?i)(?:^|[\s;&|])ruff\s+check(?:\s|$)',
            'hook': r'(?i)\b(?:ruff|ruff-pre-commit)\b',
            'filenames': set(),
        },
        'flake8': {
            'config': [r'(?im)^\s*\[flake8\]\s*$', r'(?im)^\s*\[tool\.flake8\]\s*$'],
            'dep': r'(?im)^\s*flake8(?:\[[^\]]+\])?\s*(?:[=<>!~]|$)',
            'cmd': r'(?i)(?:^|[\s;&|])flake8(?:\s|$)',
            'hook': r'(?i)\bflake8\b',
            'filenames': {'.flake8'},
        },
        'pylint': {
            'config': [r'(?im)^\s*\[tool\.pylint(?:\.|\])', r'(?im)^\s*\[(?:MASTER|MAIN)\]\s*$'],
            'dep': r'(?im)^\s*pylint(?:\[[^\]]+\])?\s*(?:[=<>!~]|$)',
            'cmd': r'(?i)(?:^|[\s;&|])pylint(?:\s|$)',
            'hook': r'(?i)\bpylint\b',
            'filenames': {'.pylintrc', 'pylintrc'},
        },
    },
    'type': {
        'mypy': {
            'config': [r'(?im)^\s*\[mypy(?:-|\])', r'(?im)^\s*\[tool\.mypy\]\s*$'],
            'dep': r'(?im)^\s*mypy(?:\[[^\]]+\])?\s*(?:[=<>!~]|$)',
            'cmd': r'(?i)(?:^|[\s;&|])mypy(?:\s|$)',
            'hook': r'(?i)\bmypy\b',
            'filenames': {'mypy.ini', '.mypy.ini'},
        },
        'pyright': {
            'config': [r'(?im)^\s*\[tool\.pyright\]\s*$'],
            'dep': r'(?im)^\s*(?:pyright|basedpyright)\s*(?:[=<>!~]|$)',
            'cmd': r'(?i)(?:^|[\s;&|])(?:pyright|basedpyright)(?:\s|$)',
            'hook': r'(?i)\b(?:pyright|basedpyright)\b',
            'filenames': {'pyrightconfig.json'},
        },
    },
    'format': {
        'black': {
            'config': [r'(?im)^\s*\[tool\.black\]\s*$'],
            'dep': r'(?im)^\s*black(?:\[[^\]]+\])?\s*(?:[=<>!~]|$)',
            'cmd': r'(?i)(?:^|[\s;&|])black\b[^\n]*--check\b',
            'hook': r'(?i)\bblack\b',
            'filenames': set(),
        },
        'ruff_format': {
            'config': [r'(?im)^\s*\[tool\.ruff\.format\]\s*$'],
            'dep': r'(?im)^\s*ruff(?:\[[^\]]+\])?\s*(?:[=<>!~]|$)',
            'cmd': r'(?i)(?:^|[\s;&|])ruff\s+format\b[^\n]*--check\b',
            'hook': r'(?i)\bruff-format\b',
            'filenames': set(),
        },
    },
}

AI_PATTERNS = {
    'chatgpt': [
        r'(?i)\b(?:built|created|developed|generated|written|coded|made)\s+(?:entirely\s+|fully\s+|partly\s+|mostly\s+)?(?:with|using|by)\s+chatgpt\b',
        r'(?i)\bchatgpt[- ]generated\b',
    ],
    'claude': [
        r'(?i)\b(?:built|created|developed|generated|written|coded|made)\s+(?:entirely\s+|fully\s+|partly\s+|mostly\s+)?(?:with|using|by)\s+(?:claude|claude\s+code)\b',
        r'(?i)\bclaude(?:\s+code)?[- ]generated\b',
    ],
    'copilot': [
        r'(?i)\b(?:built|created|developed|generated|written|coded|made)\s+(?:entirely\s+|fully\s+|partly\s+|mostly\s+)?(?:with|using|by)\s+(?:github\s+)?copilot\b',
        r'(?i)\bcopilot[- ]generated\b',
    ],
    'cursor': [
        r'(?i)\b(?:built|created|developed|generated|written|coded|made)\s+(?:entirely\s+|fully\s+|partly\s+|mostly\s+)?(?:with|using|by)\s+cursor\b',
    ],
    'vibe_coding': [r'(?i)\bvibe[- ]cod(?:ed|ing)\b', r'(?i)\bfully\s+vibecoded\b'],
    'generic_ai': [
        r'(?i)\b(?:entirely|fully|mostly)\s+(?:ai|artificial intelligence)[- ]generated\b',
        r'(?i)\b(?:built|created|developed|generated|written|coded|made)\s+(?:entirely\s+|fully\s+|mostly\s+)?(?:with|using|by)\s+(?:generative\s+)?ai\b',
    ],
}


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot', type=int, default=None, metavar='N', help='analyze at most N repositories per cohort')
    p.add_argument('--seed', type=int, default=20260825)
    return p.parse_args()


def safe_read(path: Path) -> str:
    try:
        if not path.is_file() or path.stat().st_size > MAX_TEXT_BYTES:
            return ''
        return path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return ''


def iter_files(root: Path) -> Iterable[Path]:
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS]
        base = Path(current)
        for name in files:
            yield base / name


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def is_ci(r: str) -> bool:
    low = r.lower()
    return any(low.startswith(x) or low == x for x in CI_PREFIXES)


def is_precommit(r: str) -> bool:
    return r.lower() == PRECOMMIT_PATH


def dotted_module(r: str):
    p = PurePosixPath(r)
    if p.suffix.lower() != '.py':
        return None
    parts = list(p.with_suffix('').parts)
    if parts and parts[-1] == '__init__':
        parts = parts[:-1]
    if not parts or any(not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', x) for x in parts):
        return None
    return '.'.join(parts)


def aliases_for_module(r: str) -> set[str]:
    mod = dotted_module(r)
    if not mod:
        return set()
    parts = mod.split('.')
    out = set()
    for i in range(len(parts)):
        candidate = '.'.join(parts[i:])
        out.add(candidate)
        out.add(candidate.split('.')[0])
    out -= {'src', 'lib', 'python', 'app', 'test', 'tests', '__init__', 'setup', 'conftest', 'main', 'utils', 'common'}
    return out


def imports_from_tree(tree: ast.AST) -> set[str]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name)
                out.add(a.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            out.add(node.module.split('.')[0])
    return out


def called_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ''


def inspect_test(path: Path, prod_aliases: set[str]) -> dict:
    out = dict(ast_ok=0, production_reference=0, explicit_oracle=0,
               assert_count=0, unittest_assert_count=0,
               exception_oracle_count=0, mock_oracle_count=0)
    try:
        if path.stat().st_size > MAX_PYTHON_BYTES:
            return out
        tree = ast.parse(path.read_text(encoding='utf-8', errors='replace'))
    except (OSError, SyntaxError, ValueError):
        return out

    out['ast_ok'] = 1
    imports = imports_from_tree(tree)
    for imp in imports:
        if imp in prod_aliases or any(imp.startswith(a + '.') or a.startswith(imp + '.') for a in prod_aliases):
            out['production_reference'] = 1
            break

    has_oracle = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            out['assert_count'] += 1
            has_oracle = True
        elif isinstance(node, ast.Call):
            name = called_name(node)
            if name in UNITTEST_ASSERTS:
                out['unittest_assert_count'] += 1
                has_oracle = True
                if name.startswith('assertRaise') or name.startswith('assertWarn'):
                    out['exception_oracle_count'] += 1
            elif name in {'raises', 'warns'}:
                out['exception_oracle_count'] += 1
                has_oracle = True
            if name in MOCK_ASSERTS:
                out['mock_oracle_count'] += 1
                has_oracle = True
    out['explicit_oracle'] = int(has_oracle)
    return out


def text_candidates(root: Path, files: list[Path]) -> list[tuple[str, str]]:
    out = []
    for p in files:
        r = rel(p, root)
        low = p.name.lower()
        if low in CONFIG_NAMES or low in DEPENDENCY_NAMES or is_ci(r) or is_precommit(r) or README_RE.match(low):
            s = safe_read(p)
            if s:
                out.append((r, s))
    return out


def detect_family(texts, family):
    configured, declared, ci, precommit = set(), set(), set(), set()
    for r, text in texts:
        filename = PurePosixPath(r.lower()).name
        for tool, spec in family.items():
            if filename in spec['filenames'] or any(re.search(p, text) for p in spec['config']):
                configured.add(tool)
            if filename in DEPENDENCY_NAMES and re.search(spec['dep'], text):
                declared.add(tool)
            if is_ci(r) and re.search(spec['cmd'], text):
                ci.add(tool)
            if is_precommit(r) and re.search(spec['hook'], text):
                precommit.add(tool)
    all_tools = configured | declared | ci | precommit
    return {
        'tools': sorted(all_tools), 'configured': sorted(configured),
        'declared': sorted(declared), 'ci': sorted(ci),
        'precommit': sorted(precommit),
    }


def detect_ai(description: str, texts):
    tools, sources = set(), set()
    candidates = []
    if description and description.lower() != 'nan':
        candidates.append(('description', description))
    for r, text in texts:
        if README_RE.match(PurePosixPath(r).name):
            candidates.append((r, text))
    for source, text in candidates:
        for tool, patterns in AI_PATTERNS.items():
            if any(re.search(p, text) for p in patterns):
                tools.add(tool)
                sources.add(source)
    return tools, sources


def analyze_repo(repo, artifacts):
    start = time.perf_counter()
    root = Path(str(repo.clone_path))
    out = {
        'repo_id': int(repo.repo_id), 'cohort': str(repo.cohort),
        'full_name': str(repo.full_name), 'quality_status': 'ok', 'error_message': '',
        'test_artifact_count_analyzed': 0, 'test_artifact_ast_ok_count': 0,
        'test_files_with_production_reference': 0, 'test_files_with_production_reference_ratio': 0.0,
        'has_production_reference_test': 0,
        'test_files_with_explicit_oracle': 0, 'test_files_with_explicit_oracle_ratio': 0.0,
        'has_explicit_oracle': 0, 'explicit_assert_count': 0,
        'explicit_unittest_assert_count': 0, 'explicit_exception_oracle_count': 0,
        'explicit_mock_oracle_count': 0,
        'has_linting': 0, 'lint_tools': '', 'lint_configured': 0, 'lint_declared': 0, 'lint_ci': 0, 'lint_precommit': 0,
        'has_type_checking': 0, 'type_tools': '', 'type_configured': 0, 'type_declared': 0, 'type_ci': 0, 'type_precommit': 0,
        'has_formatting': 0, 'format_tools': '', 'format_configured': 0, 'format_declared': 0, 'format_ci': 0, 'format_precommit': 0,
        'has_other_qa': 0, 'has_other_qa_automated': 0,
        'ai_explicit': 0, 'ai_explicit_tools': '', 'ai_explicit_sources': '',
        'analysis_elapsed_seconds': 0.0,
    }
    if not root.exists():
        out['quality_status'] = 'clone_missing'
        out['error_message'] = f'clone path does not exist: {root}'
        out['analysis_elapsed_seconds'] = time.perf_counter() - start
        return out

    try:
        files = list(iter_files(root))
        test_paths = set()
        if artifacts is not None and not artifacts.empty:
            test_paths = {str(x).replace('\\', '/') for x in artifacts['path'].dropna()}

        prod_aliases = set()
        for p in files:
            r = rel(p, root)
            if p.suffix.lower() == '.py' and r not in test_paths:
                prod_aliases |= aliases_for_module(r)

        n_test = n_ast = n_prod = n_oracle = 0
        n_assert = n_unittest = n_exc = n_mock = 0
        for r in sorted(test_paths):
            p = root / r
            if not p.exists() or p.suffix.lower() != '.py':
                continue
            n_test += 1
            x = inspect_test(p, prod_aliases)
            n_ast += x['ast_ok']
            n_prod += x['production_reference']
            n_oracle += x['explicit_oracle']
            n_assert += x['assert_count']
            n_unittest += x['unittest_assert_count']
            n_exc += x['exception_oracle_count']
            n_mock += x['mock_oracle_count']

        out.update({
            'test_artifact_count_analyzed': n_test,
            'test_artifact_ast_ok_count': n_ast,
            'test_files_with_production_reference': n_prod,
            'test_files_with_production_reference_ratio': n_prod / n_test if n_test else 0.0,
            'has_production_reference_test': int(n_prod > 0),
            'test_files_with_explicit_oracle': n_oracle,
            'test_files_with_explicit_oracle_ratio': n_oracle / n_test if n_test else 0.0,
            'has_explicit_oracle': int(n_oracle > 0),
            'explicit_assert_count': n_assert,
            'explicit_unittest_assert_count': n_unittest,
            'explicit_exception_oracle_count': n_exc,
            'explicit_mock_oracle_count': n_mock,
        })

        texts = text_candidates(root, files)
        for prefix, family in [('lint', TOOL_FAMILIES['lint']), ('type', TOOL_FAMILIES['type']), ('format', TOOL_FAMILIES['format'])]:
            d = detect_family(texts, family)
            out[f'has_{"type_checking" if prefix == "type" else "formatting" if prefix == "format" else "linting"}'] = int(bool(d['tools']))
            out[f'{prefix}_tools'] = '|'.join(d['tools'])
            out[f'{prefix}_configured'] = int(bool(d['configured']))
            out[f'{prefix}_declared'] = int(bool(d['declared']))
            out[f'{prefix}_ci'] = int(bool(d['ci']))
            out[f'{prefix}_precommit'] = int(bool(d['precommit']))

        out['has_other_qa'] = int(out['has_linting'] or out['has_type_checking'] or out['has_formatting'])
        out['has_other_qa_automated'] = int(
            out['lint_ci'] or out['lint_precommit'] or
            out['type_ci'] or out['type_precommit'] or
            out['format_ci'] or out['format_precommit']
        )

        ai_tools, ai_sources = detect_ai(str(getattr(repo, 'description', '') or ''), texts)
        out['ai_explicit'] = int(bool(ai_tools))
        out['ai_explicit_tools'] = '|'.join(sorted(ai_tools))
        out['ai_explicit_sources'] = '|'.join(sorted(ai_sources))

    except Exception as exc:
        out['quality_status'] = 'inspection_failed'
        out['error_message'] = repr(exc)

    out['analysis_elapsed_seconds'] = time.perf_counter() - start
    return out


def summary(df):
    print('\n' + '=' * 72)
    print('04 QUALITY EXTENSION SUMMARY')
    print('=' * 72)
    for cohort in COHORTS:
        g = df[df['cohort'] == cohort]
        if g.empty:
            continue
        print(f'\n{cohort}\nprocessed: {len(g):,}\nstatus:')
        print(g['quality_status'].value_counts(dropna=False).to_string())
        ok = g[g['quality_status'] == 'ok']
        tested = ok[ok['test_artifact_count_analyzed'] > 0]
        if len(tested):
            print('\nRQ2 additions (repositories with analyzed test artifacts):')
            for label, col in [
                ('production-code reference', 'has_production_reference_test'),
                ('explicit oracle', 'has_explicit_oracle'),
            ]:
                n = int(tested[col].sum())
                print(f'{label:<28} {n:,}/{len(tested):,} ({n/len(tested)*100:.2f}%)')
        print('\nRQ4 / exploratory additions (all repositories):')
        for label, col in [
            ('linting', 'has_linting'), ('type checking', 'has_type_checking'),
            ('formatting', 'has_formatting'), ('other QA any', 'has_other_qa'),
            ('other QA automated', 'has_other_qa_automated'), ('AI use explicit', 'ai_explicit'),
        ]:
            n = int(ok[col].sum())
            print(f'{label:<28} {n:,}/{len(ok):,} ({n/len(ok)*100:.2f}%)')
        print(f"median time: {ok['analysis_elapsed_seconds'].median():.4f}s/repo")


def main():
    a = args()
    for p in (REPO_METRICS, TEST_ARTIFACTS, SELECTED):
        if not p.exists():
            raise FileNotFoundError(p)

    metrics = pd.read_csv(REPO_METRICS)
    artifacts = pd.read_csv(TEST_ARTIFACTS)
    selected = pd.read_csv(SELECTED)
    metrics = metrics[metrics['analysis_status'] == 'ok'].copy()

    if 'description' in selected.columns:
        metrics = metrics.merge(
            selected[['repo_id', 'description']].drop_duplicates('repo_id'),
            on='repo_id', how='left', validate='one_to_one'
        )

    artifacts_by_repo = {int(k): v.copy() for k, v in artifacts.groupby('repo_id', sort=False)}

    if a.pilot is not None:
        if a.pilot <= 0:
            raise ValueError('--pilot must be > 0')
        groups = []
        for cohort in COHORTS:
            g = metrics[metrics['cohort'] == cohort]
            if len(g):
                groups.append(g.sample(n=min(a.pilot, len(g)), random_state=a.seed))
        metrics = pd.concat(groups, ignore_index=True) if groups else metrics.iloc[:0].copy()

    metrics['cohort'] = pd.Categorical(metrics['cohort'], categories=COHORTS, ordered=True)
    metrics = metrics.sort_values(['cohort', 'repo_id'], kind='stable').reset_index(drop=True)
    target = PILOT_OUTPUT if a.pilot is not None else OUTPUT

    print('=' * 72)
    print('04 QUALITY EXTENSION')
    print('=' * 72)
    print(f'repositories: {len(metrics):,}')
    print(f'pilot:        {a.pilot if a.pilot is not None else "no"}')
    print(f'output:       {target}\n')

    rows = []
    started = time.time()
    for i, repo in enumerate(metrics.itertuples(index=False), 1):
        row = analyze_repo(repo, artifacts_by_repo.get(int(repo.repo_id)))
        rows.append(row)
        if i <= 20 or i % 250 == 0 or i == len(metrics):
            elapsed = time.time() - started
            rate = i / elapsed if elapsed else 0
            eta = (len(metrics) - i) / rate if rate else 0
            print(f'[{i:>6,d}/{len(metrics):,}] {str(repo.cohort):<12} {str(repo.full_name)[:42]:<42} '
                  f'{row["quality_status"]:<17} {row["analysis_elapsed_seconds"]:.3f}s ETA {eta/60:.1f}m', flush=True)

    out = pd.DataFrame(rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    summary(out)
    print(f'\nOutput: {target}')
    if a.pilot is not None:
        print('Pilot output is separate; run without --pilot for the full analysis.')


if __name__ == '__main__':
    main()
