from __future__ import annotations

import argparse
import ast
import os
import re
import sqlite3
import time
import warnings

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yaml


CONFIG_PATH = Path("config.yaml")

COHORTS = [
    "pre_ai",
    "contemporary",
]


# ============================================================
# Configuration
# ============================================================

def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# Data classes
# ============================================================

@dataclass
class TestArtifactResult:
    repo_id: int
    path: str

    loc: int
    size_bytes: int

    detection_reason: str

    is_init: int
    is_standard_named: int
    is_test_prefix: int
    is_test_suffix: int
    is_test_directory: int

    ast_status: str

    test_function_count: int
    test_class_count: int

    assert_statement_count: int
    unittest_assertion_count: int

    imports_pytest: int
    imports_unittest: int
    imports_doctest: int

    uses_fixture: int
    uses_parametrize: int
    uses_mock: int
    uses_monkeypatch: int
    uses_exception_testing: int

    has_strong_evidence: int


@dataclass
class RepositoryResult:
    repo_id: int
    cohort: str
    full_name: str
    clone_path: str

    analysis_status: str
    analysis_elapsed_seconds: float
    error_message: str | None

    # --------------------------------------------------------
    # Project size
    # --------------------------------------------------------

    python_file_count: int
    python_loc: int

    production_python_file_count: int
    production_python_loc: int

    # --------------------------------------------------------
    # Test artifacts
    # --------------------------------------------------------

    test_artifact_file_count: int
    test_artifact_loc: int
    test_artifact_loc_ratio: float | None

    standard_named_test_file_count: int
    test_directory_only_file_count: int
    init_test_file_count: int

    # --------------------------------------------------------
    # Final test-presence definitions
    #
    # broad:
    #   standard name OR test/tests directory
    #
    # conventional:
    #   strong evidence OR standard test filename
    #
    # strict:
    #   strong code-level evidence
    # --------------------------------------------------------

    has_tests_broad: int
    has_tests_conventional: int
    has_tests_strict: int

    # Weak category applies only when:
    #
    # broad == 1
    # strict == 0
    #
    # Values:
    #   none
    #   standard_named_test_file
    #   test_directory_only
    #   init_only
    #   syntax_error_only
    #   other
    weak_evidence_category: str

    # --------------------------------------------------------
    # Test contents
    # --------------------------------------------------------

    test_function_count: int
    test_class_count: int

    assert_statement_count: int
    unittest_assertion_count: int

    ast_parse_failure_count: int
    ast_skipped_large_count: int

    # --------------------------------------------------------
    # Framework / technique evidence in test files
    # --------------------------------------------------------

    testcode_uses_pytest: int
    testcode_uses_unittest: int
    testcode_uses_doctest: int

    uses_fixture: int
    uses_parametrize: int
    uses_mock: int
    uses_monkeypatch: int
    uses_exception_testing: int

    # --------------------------------------------------------
    # Project test/config files
    # --------------------------------------------------------

    has_conftest: int
    has_pytest_ini: int
    has_tox_ini: int
    has_noxfile: int

    has_pyproject: int
    has_setup_cfg: int
    has_setup_py: int
    has_requirements: int
    has_pipfile: int

    # --------------------------------------------------------
    # Dependency/config declarations
    # --------------------------------------------------------

    declares_pytest: int
    declares_pytest_cov: int
    declares_coverage: int
    declares_tox: int
    declares_nox: int
    declares_hypothesis: int

    # --------------------------------------------------------
    # CI
    # --------------------------------------------------------

    has_ci: int

    has_github_actions: int
    has_travis_ci: int
    has_circle_ci: int
    has_gitlab_ci: int

    has_ci_test_execution: int

    ci_uses_pytest: int
    ci_uses_unittest: int
    ci_uses_tox: int
    ci_uses_nox: int

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    has_coverage_config: int
    has_coverage_tooling: int


# ============================================================
# SQLite
# ============================================================

def connect_database(
    path: Path,
) -> sqlite3.Connection:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(path)

    conn.execute(
        "PRAGMA journal_mode=WAL"
    )

    conn.execute(
        "PRAGMA synchronous=NORMAL"
    )

    create_tables(conn)

    return conn


def create_tables(
    conn: sqlite3.Connection,
):

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS repository_metrics (

            repo_id INTEGER PRIMARY KEY,

            cohort TEXT NOT NULL,
            full_name TEXT NOT NULL,
            clone_path TEXT NOT NULL,

            analysis_status TEXT NOT NULL,
            analysis_elapsed_seconds REAL,
            error_message TEXT,

            python_file_count INTEGER,
            python_loc INTEGER,

            production_python_file_count INTEGER,
            production_python_loc INTEGER,

            test_artifact_file_count INTEGER,
            test_artifact_loc INTEGER,
            test_artifact_loc_ratio REAL,

            standard_named_test_file_count INTEGER,
            test_directory_only_file_count INTEGER,
            init_test_file_count INTEGER,

            has_tests_broad INTEGER,
            has_tests_conventional INTEGER,
            has_tests_strict INTEGER,

            weak_evidence_category TEXT,

            test_function_count INTEGER,
            test_class_count INTEGER,

            assert_statement_count INTEGER,
            unittest_assertion_count INTEGER,

            ast_parse_failure_count INTEGER,
            ast_skipped_large_count INTEGER,

            testcode_uses_pytest INTEGER,
            testcode_uses_unittest INTEGER,
            testcode_uses_doctest INTEGER,

            uses_fixture INTEGER,
            uses_parametrize INTEGER,
            uses_mock INTEGER,
            uses_monkeypatch INTEGER,
            uses_exception_testing INTEGER,

            has_conftest INTEGER,
            has_pytest_ini INTEGER,
            has_tox_ini INTEGER,
            has_noxfile INTEGER,

            has_pyproject INTEGER,
            has_setup_cfg INTEGER,
            has_setup_py INTEGER,
            has_requirements INTEGER,
            has_pipfile INTEGER,

            declares_pytest INTEGER,
            declares_pytest_cov INTEGER,
            declares_coverage INTEGER,
            declares_tox INTEGER,
            declares_nox INTEGER,
            declares_hypothesis INTEGER,

            has_ci INTEGER,

            has_github_actions INTEGER,
            has_travis_ci INTEGER,
            has_circle_ci INTEGER,
            has_gitlab_ci INTEGER,

            has_ci_test_execution INTEGER,

            ci_uses_pytest INTEGER,
            ci_uses_unittest INTEGER,
            ci_uses_tox INTEGER,
            ci_uses_nox INTEGER,

            has_coverage_config INTEGER,
            has_coverage_tooling INTEGER,

            processed_at TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS test_artifacts (

            repo_id INTEGER NOT NULL,
            path TEXT NOT NULL,

            loc INTEGER,
            size_bytes INTEGER,

            detection_reason TEXT,

            is_init INTEGER,
            is_standard_named INTEGER,
            is_test_prefix INTEGER,
            is_test_suffix INTEGER,
            is_test_directory INTEGER,

            ast_status TEXT,

            test_function_count INTEGER,
            test_class_count INTEGER,

            assert_statement_count INTEGER,
            unittest_assertion_count INTEGER,

            imports_pytest INTEGER,
            imports_unittest INTEGER,
            imports_doctest INTEGER,

            uses_fixture INTEGER,
            uses_parametrize INTEGER,
            uses_mock INTEGER,
            uses_monkeypatch INTEGER,
            uses_exception_testing INTEGER,

            has_strong_evidence INTEGER,

            PRIMARY KEY (
                repo_id,
                path
            )
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_repository_metrics_cohort
        ON repository_metrics(
            cohort
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_repository_metrics_test_defs
        ON repository_metrics(
            has_tests_broad,
            has_tests_conventional,
            has_tests_strict
        )
        """
    )

    conn.commit()


# ============================================================
# Basic file helpers
# ============================================================

def count_lines(
    path: Path,
) -> int:

    count = 0

    try:
        with path.open("rb") as f:
            for _ in f:
                count += 1

    except (
        OSError,
        PermissionError,
    ):
        return 0

    return count


def read_text_file(
    path: Path,
    max_bytes: int,
) -> str | None:

    try:

        if path.stat().st_size > max_bytes:
            return None

        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    except (
        OSError,
        PermissionError,
    ):

        return None


# ============================================================
# Test-artifact detection
# ============================================================

def classify_test_artifact(
    relative_path: Path,
) -> dict:

    filename = (
        relative_path.name.lower()
    )

    parent_parts = [
        p.lower()
        for p
        in relative_path.parts[:-1]
    ]

    is_test_prefix = (
        filename.startswith("test_")
        and filename.endswith(".py")
    )

    is_test_suffix = (
        filename.endswith("_test.py")
    )

    is_test_directory = (
        "test" in parent_parts
        or "tests" in parent_parts
    )

    is_init = (
        filename == "__init__.py"
    )

    is_standard_named = (
        is_test_prefix
        or is_test_suffix
    )

    reasons = []

    if is_test_prefix:
        reasons.append(
            "test_prefix"
        )

    if is_test_suffix:
        reasons.append(
            "test_suffix"
        )

    if is_test_directory:
        reasons.append(
            "test_directory"
        )

    # conftest.py is test infrastructure,
    # not counted as a test artifact itself.
    is_candidate = (
        filename != "conftest.py"
        and (
            is_standard_named
            or is_test_directory
        )
    )

    return {
        "is_candidate":
            is_candidate,

        "is_init":
            is_init,

        "is_standard_named":
            is_standard_named,

        "is_test_prefix":
            is_test_prefix,

        "is_test_suffix":
            is_test_suffix,

        "is_test_directory":
            is_test_directory,

        "detection_reason":
            "|".join(reasons),
    }


# ============================================================
# AST utilities
# ============================================================

def dotted_name(
    node,
) -> str:

    if isinstance(
        node,
        ast.Name,
    ):
        return node.id

    if isinstance(
        node,
        ast.Attribute,
    ):

        left = dotted_name(
            node.value
        )

        if left:
            return (
                left
                + "."
                + node.attr
            )

        return node.attr

    if isinstance(
        node,
        ast.Call,
    ):
        return dotted_name(
            node.func
        )

    return ""


class TestAstVisitor(
    ast.NodeVisitor
):

    def __init__(
        self,
    ):

        self.imports_pytest = False
        self.imports_unittest = False
        self.imports_doctest = False

        self.test_function_count = 0
        self.test_class_count = 0

        self.assert_statement_count = 0
        self.unittest_assertion_count = 0

        self.uses_fixture = False
        self.uses_parametrize = False
        self.uses_mock = False
        self.uses_monkeypatch = False
        self.uses_exception_testing = False

    def visit_Import(
        self,
        node: ast.Import,
    ):

        for alias in node.names:

            name = alias.name

            if (
                name == "pytest"
                or name.startswith("pytest.")
            ):
                self.imports_pytest = True

            if (
                name == "unittest"
                or name.startswith("unittest.")
            ):
                self.imports_unittest = True

            if (
                name == "doctest"
                or name.startswith("doctest.")
            ):
                self.imports_doctest = True

            if (
                name == "mock"
                or name.startswith("mock.")
                or name.startswith(
                    "unittest.mock"
                )
            ):
                self.uses_mock = True

        self.generic_visit(node)

    def visit_ImportFrom(
        self,
        node: ast.ImportFrom,
    ):

        module = (
            node.module
            or ""
        )

        if (
            module == "pytest"
            or module.startswith("pytest.")
        ):
            self.imports_pytest = True

        if (
            module == "unittest"
            or module.startswith("unittest.")
        ):
            self.imports_unittest = True

        if (
            module == "doctest"
            or module.startswith("doctest.")
        ):
            self.imports_doctest = True

        if (
            module == "mock"
            or module.startswith("mock.")
            or module.startswith(
                "unittest.mock"
            )
        ):
            self.uses_mock = True

        self.generic_visit(node)

    def _visit_function(
        self,
        node,
    ):

        if node.name.startswith(
            "test"
        ):
            self.test_function_count += 1

        for decorator in (
            node.decorator_list
        ):

            name = dotted_name(
                decorator
            )

            lname = name.lower()

            if (
                lname.endswith(".fixture")
                or lname == "fixture"
            ):
                self.uses_fixture = True

            if (
                "parametrize"
                in lname
            ):
                self.uses_parametrize = True

        args = (
            list(node.args.args)
            + list(
                node.args.kwonlyargs
            )
        )

        for arg in args:

            if (
                arg.arg
                == "monkeypatch"
            ):
                self.uses_monkeypatch = True

        self.generic_visit(node)

    def visit_FunctionDef(
        self,
        node: ast.FunctionDef,
    ):
        self._visit_function(node)

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ):
        self._visit_function(node)

    def visit_ClassDef(
        self,
        node: ast.ClassDef,
    ):

        is_test_class = (
            node.name.startswith(
                "Test"
            )
        )

        for base in node.bases:

            name = dotted_name(
                base
            )

            if name.endswith(
                "TestCase"
            ):

                is_test_class = True
                self.imports_unittest = True

        if is_test_class:
            self.test_class_count += 1

        self.generic_visit(node)

    def visit_Assert(
        self,
        node: ast.Assert,
    ):

        self.assert_statement_count += 1

        self.generic_visit(node)

    def visit_Call(
        self,
        node: ast.Call,
    ):

        name = dotted_name(
            node.func
        )

        lname = name.lower()

        final_name = (
            name.split(".")[-1]
            if name
            else ""
        )

        # unittest assertions
        if (
            final_name.startswith(
                "assert"
            )
            and (
                "." in name
                or lname.startswith(
                    "assert"
                )
            )
        ):
            self.unittest_assertion_count += 1

        # exception testing
        if (
            "pytest.raises"
            in lname
            or final_name
            == "assertRaises"
        ):
            self.uses_exception_testing = True

        if (
            "parametrize"
            in lname
        ):
            self.uses_parametrize = True

        if (
            "mock"
            in lname
            or final_name == "patch"
        ):
            self.uses_mock = True

        self.generic_visit(node)


def analyze_test_artifact_ast(
    path: Path,
    max_ast_bytes: int,
) -> dict:

    result = {
        "ast_status":
            "ok",

        "test_function_count":
            0,

        "test_class_count":
            0,

        "assert_statement_count":
            0,

        "unittest_assertion_count":
            0,

        "imports_pytest":
            0,

        "imports_unittest":
            0,

        "imports_doctest":
            0,

        "uses_fixture":
            0,

        "uses_parametrize":
            0,

        "uses_mock":
            0,

        "uses_monkeypatch":
            0,

        "uses_exception_testing":
            0,

        "has_strong_evidence":
            0,
    }

    try:
        size = (
            path.stat().st_size
        )

    except OSError:

        result[
            "ast_status"
        ] = "read_error"

        return result

    if size > max_ast_bytes:

        result[
            "ast_status"
        ] = "skipped_large"

        return result

    try:

        source = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore",
                SyntaxWarning,
            )

            tree = ast.parse(
                source
            )

    except SyntaxError:

        result[
            "ast_status"
        ] = "syntax_error"

        return result

    except Exception:

        result[
            "ast_status"
        ] = "parse_error"

        return result

    visitor = TestAstVisitor()

    visitor.visit(tree)

    result.update({
        "test_function_count":
            visitor.test_function_count,

        "test_class_count":
            visitor.test_class_count,

        "assert_statement_count":
            visitor.assert_statement_count,

        "unittest_assertion_count":
            visitor.unittest_assertion_count,

        "imports_pytest":
            int(
                visitor.imports_pytest
            ),

        "imports_unittest":
            int(
                visitor.imports_unittest
            ),

        "imports_doctest":
            int(
                visitor.imports_doctest
            ),

        "uses_fixture":
            int(
                visitor.uses_fixture
            ),

        "uses_parametrize":
            int(
                visitor.uses_parametrize
            ),

        "uses_mock":
            int(
                visitor.uses_mock
            ),

        "uses_monkeypatch":
            int(
                visitor.uses_monkeypatch
            ),

        "uses_exception_testing":
            int(
                visitor
                .uses_exception_testing
            ),
    })

    result[
        "has_strong_evidence"
    ] = int(
        result[
            "test_function_count"
        ] > 0
        or result[
            "test_class_count"
        ] > 0
        or result[
            "assert_statement_count"
        ] > 0
        or result[
            "unittest_assertion_count"
        ] > 0
        or result[
            "imports_pytest"
        ] == 1
        or result[
            "imports_unittest"
        ] == 1
    )

    return result


# ============================================================
# Dependency / CI helpers
# ============================================================

def is_requirements_file(
    filename: str,
) -> bool:

    filename = filename.lower()

    return (
        filename.startswith(
            "requirements"
        )
        and (
            filename.endswith(
                ".txt"
            )
            or filename
            == "requirements"
        )
    )


CI_TEST_PATTERNS = [
    re.compile(
        r"(^|\s)pytest(\s|$)",
        re.IGNORECASE,
    ),

    re.compile(
        r"python\s+-m\s+pytest",
        re.IGNORECASE,
    ),

    re.compile(
        r"python(?:3)?\s+-m\s+unittest",
        re.IGNORECASE,
    ),

    re.compile(
        r"unittest\s+discover",
        re.IGNORECASE,
    ),

    re.compile(
        r"(^|\s)tox(\s|$)",
        re.IGNORECASE,
    ),

    re.compile(
        r"(^|\s)nox(\s|$)",
        re.IGNORECASE,
    ),
]


# ============================================================
# Weak evidence classification
# ============================================================

def classify_weak_evidence(
    artifacts: list[TestArtifactResult],
    has_strict: bool,
) -> str:

    if not artifacts:
        return "none"

    if has_strict:
        return "none"

    if all(
        a.ast_status
        == "syntax_error"
        for a in artifacts
    ):
        return "syntax_error_only"

    if all(
        a.is_init == 1
        for a in artifacts
    ):
        return "init_only"

    if any(
        a.is_standard_named == 1
        for a in artifacts
    ):
        return (
            "standard_named_test_file"
        )

    if all(
        a.is_test_directory == 1
        for a in artifacts
    ):
        return (
            "test_directory_only"
        )

    return "other"


# ============================================================
# Analyze one repository
# ============================================================

def analyze_repository(
    row: dict,
    config: dict,
) -> tuple[
    RepositoryResult,
    list[TestArtifactResult],
]:

    started = time.time()

    repo_id = int(
        row["repo_id"]
    )

    cohort = str(
        row["cohort"]
    )

    full_name = str(
        row["full_name"]
    )

    repo_path = Path(
        row["clone_path"]
    )

    analysis_config = (
        config["analysis"]
    )

    exclude_dirs = set(
        analysis_config[
            "exclude_dirs"
        ]
    )

    max_ast_bytes = int(
        float(
            analysis_config.get(
                "max_ast_file_mb",
                5,
            )
        )
        * 1024
        * 1024
    )

    max_text_bytes = int(
        float(
            analysis_config.get(
                "max_text_file_mb",
                5,
            )
        )
        * 1024
        * 1024
    )

    if not repo_path.exists():

        return (
            empty_repository_result(
                repo_id=repo_id,
                cohort=cohort,
                full_name=full_name,
                clone_path=str(
                    repo_path
                ),
                status=(
                    "missing_repository"
                ),
                error_message=(
                    "Repository directory "
                    "does not exist."
                ),
                elapsed=(
                    time.time()
                    - started
                ),
            ),
            [],
        )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    artifacts = []

    python_file_count = 0
    python_loc = 0

    production_python_file_count = 0
    production_python_loc = 0

    # config / tools
    has_conftest = False
    has_pytest_ini = False
    has_tox_ini = False
    has_noxfile = False

    has_pyproject = False
    has_setup_cfg = False
    has_setup_py = False
    has_requirements = False
    has_pipfile = False

    declares_pytest = False
    declares_pytest_cov = False
    declares_coverage = False
    declares_tox = False
    declares_nox = False
    declares_hypothesis = False

    has_github_actions = False
    has_travis_ci = False
    has_circle_ci = False
    has_gitlab_ci = False

    has_ci_test_execution = False

    ci_uses_pytest = False
    ci_uses_unittest = False
    ci_uses_tox = False
    ci_uses_nox = False

    has_coverage_config = False
    has_coverage_tooling = False

    # --------------------------------------------------------
    # Walk repository
    # --------------------------------------------------------

    for root, dirs, files in os.walk(
        repo_path,
        followlinks=False,
    ):

        kept_dirs = []

        for dirname in dirs:

            path = (
                Path(root)
                / dirname
            )

            if dirname in exclude_dirs:
                continue

            if path.is_symlink():
                continue

            kept_dirs.append(
                dirname
            )

        dirs[:] = kept_dirs

        root_path = Path(root)

        for filename in files:

            path = (
                root_path
                / filename
            )

            if path.is_symlink():
                continue

            try:
                relative_path = (
                    path.relative_to(
                        repo_path
                    )
                )

            except ValueError:
                continue

            relative_posix = (
                relative_path.as_posix()
            )

            relative_lower = (
                relative_posix.lower()
            )

            lower_name = (
                filename.lower()
            )

            # ------------------------------------------------
            # Project config
            # ------------------------------------------------

            if lower_name == "conftest.py":
                has_conftest = True

            if lower_name == "pytest.ini":
                has_pytest_ini = True

            if lower_name == "tox.ini":
                has_tox_ini = True

            if lower_name in {
                "noxfile.py",
                "noxfile",
            }:
                has_noxfile = True

            if lower_name == "pyproject.toml":
                has_pyproject = True

            if lower_name == "setup.cfg":
                has_setup_cfg = True

            if lower_name == "setup.py":
                has_setup_py = True

            if is_requirements_file(
                lower_name
            ):
                has_requirements = True

            if lower_name == "pipfile":
                has_pipfile = True

            if lower_name in {
                ".coveragerc",
                "coveragerc",
            }:
                has_coverage_config = True

            # ------------------------------------------------
            # CI files
            # ------------------------------------------------

            is_github_action = (
                relative_lower.startswith(
                    ".github/workflows/"
                )
                and (
                    lower_name.endswith(
                        ".yml"
                    )
                    or lower_name.endswith(
                        ".yaml"
                    )
                )
            )

            is_travis = (
                lower_name
                == ".travis.yml"
            )

            is_circle = (
                relative_lower
                in {
                    ".circleci/config.yml",
                    ".circleci/config.yaml",
                }
            )

            is_gitlab = (
                lower_name
                == ".gitlab-ci.yml"
            )

            is_ci_file = (
                is_github_action
                or is_travis
                or is_circle
                or is_gitlab
            )

            if is_github_action:
                has_github_actions = True

            if is_travis:
                has_travis_ci = True

            if is_circle:
                has_circle_ci = True

            if is_gitlab:
                has_gitlab_ci = True

            # ------------------------------------------------
            # Config / dependency text analysis
            # ------------------------------------------------

            should_scan_text = (
                is_ci_file
                or lower_name
                in {
                    "pyproject.toml",
                    "setup.cfg",
                    "setup.py",
                    "pipfile",
                    "pytest.ini",
                    "tox.ini",
                    "noxfile.py",
                    ".coveragerc",
                    "coveragerc",
                }
                or is_requirements_file(
                    lower_name
                )
            )

            if should_scan_text:

                text = read_text_file(
                    path,
                    max_text_bytes,
                )

                if text is not None:

                    lower_text = (
                        text.lower()
                    )

                    if re.search(
                        r"\bpytest\b",
                        lower_text,
                    ):
                        declares_pytest = True

                    if (
                        "pytest-cov"
                        in lower_text
                        or "pytest_cov"
                        in lower_text
                    ):
                        declares_pytest_cov = True

                    if re.search(
                        r"\bcoverage\b",
                        lower_text,
                    ):
                        declares_coverage = True

                    if re.search(
                        r"\btox\b",
                        lower_text,
                    ):
                        declares_tox = True

                    if re.search(
                        r"\bnox\b",
                        lower_text,
                    ):
                        declares_nox = True

                    if re.search(
                        r"\bhypothesis\b",
                        lower_text,
                    ):
                        declares_hypothesis = True

                    if (
                        "codecov"
                        in lower_text
                        or "coveralls"
                        in lower_text
                        or "pytest-cov"
                        in lower_text
                        or "coverage run"
                        in lower_text
                    ):
                        has_coverage_tooling = True

                    if is_ci_file:

                        if any(
                            pattern.search(text)
                            for pattern
                            in CI_TEST_PATTERNS
                        ):
                            has_ci_test_execution = True

                        if (
                            re.search(
                                r"(^|\s)pytest(\s|$)",
                                text,
                                flags=(
                                    re.IGNORECASE
                                ),
                            )
                            or re.search(
                                r"python\s+-m\s+pytest",
                                text,
                                flags=(
                                    re.IGNORECASE
                                ),
                            )
                        ):
                            ci_uses_pytest = True

                        if (
                            re.search(
                                r"python(?:3)?\s+-m\s+unittest",
                                text,
                                flags=(
                                    re.IGNORECASE
                                ),
                            )
                            or re.search(
                                r"unittest\s+discover",
                                text,
                                flags=(
                                    re.IGNORECASE
                                ),
                            )
                        ):
                            ci_uses_unittest = True

                        if re.search(
                            r"(^|\s)tox(\s|$)",
                            text,
                            flags=(
                                re.IGNORECASE
                            ),
                        ):
                            ci_uses_tox = True

                        if re.search(
                            r"(^|\s)nox(\s|$)",
                            text,
                            flags=(
                                re.IGNORECASE
                            ),
                        ):
                            ci_uses_nox = True

            # ------------------------------------------------
            # Python files
            # ------------------------------------------------

            if not lower_name.endswith(
                ".py"
            ):
                continue

            python_file_count += 1

            loc = count_lines(path)

            python_loc += loc

            classification = (
                classify_test_artifact(
                    relative_path
                )
            )

            if not classification[
                "is_candidate"
            ]:

                production_python_file_count += 1
                production_python_loc += loc

                continue

            # ------------------------------------------------
            # Candidate test artifact
            # ------------------------------------------------

            try:
                size_bytes = (
                    path.stat().st_size
                )

            except OSError:
                size_bytes = 0

            ast_result = (
                analyze_test_artifact_ast(
                    path,
                    max_ast_bytes,
                )
            )

            artifacts.append(
                TestArtifactResult(
                    repo_id=repo_id,

                    path=relative_posix,

                    loc=loc,

                    size_bytes=(
                        size_bytes
                    ),

                    detection_reason=(
                        classification[
                            "detection_reason"
                        ]
                    ),

                    is_init=int(
                        classification[
                            "is_init"
                        ]
                    ),

                    is_standard_named=int(
                        classification[
                            "is_standard_named"
                        ]
                    ),

                    is_test_prefix=int(
                        classification[
                            "is_test_prefix"
                        ]
                    ),

                    is_test_suffix=int(
                        classification[
                            "is_test_suffix"
                        ]
                    ),

                    is_test_directory=int(
                        classification[
                            "is_test_directory"
                        ]
                    ),

                    ast_status=(
                        ast_result[
                            "ast_status"
                        ]
                    ),

                    test_function_count=(
                        ast_result[
                            "test_function_count"
                        ]
                    ),

                    test_class_count=(
                        ast_result[
                            "test_class_count"
                        ]
                    ),

                    assert_statement_count=(
                        ast_result[
                            "assert_statement_count"
                        ]
                    ),

                    unittest_assertion_count=(
                        ast_result[
                            "unittest_assertion_count"
                        ]
                    ),

                    imports_pytest=(
                        ast_result[
                            "imports_pytest"
                        ]
                    ),

                    imports_unittest=(
                        ast_result[
                            "imports_unittest"
                        ]
                    ),

                    imports_doctest=(
                        ast_result[
                            "imports_doctest"
                        ]
                    ),

                    uses_fixture=(
                        ast_result[
                            "uses_fixture"
                        ]
                    ),

                    uses_parametrize=(
                        ast_result[
                            "uses_parametrize"
                        ]
                    ),

                    uses_mock=(
                        ast_result[
                            "uses_mock"
                        ]
                    ),

                    uses_monkeypatch=(
                        ast_result[
                            "uses_monkeypatch"
                        ]
                    ),

                    uses_exception_testing=(
                        ast_result[
                            "uses_exception_testing"
                        ]
                    ),

                    has_strong_evidence=(
                        ast_result[
                            "has_strong_evidence"
                        ]
                    ),
                )
            )

    # ========================================================
    # Repository-level aggregation
    # ========================================================

    test_artifact_file_count = (
        len(artifacts)
    )

    test_artifact_loc = sum(
        a.loc
        for a in artifacts
    )

    standard_named_count = sum(
        a.is_standard_named
        for a in artifacts
    )

    init_count = sum(
        a.is_init
        for a in artifacts
    )

    test_directory_only_count = sum(
        int(
            a.is_test_directory
            and not a.is_standard_named
        )
        for a in artifacts
    )

    test_function_count = sum(
        a.test_function_count
        for a in artifacts
    )

    test_class_count = sum(
        a.test_class_count
        for a in artifacts
    )

    assert_statement_count = sum(
        a.assert_statement_count
        for a in artifacts
    )

    unittest_assertion_count = sum(
        a.unittest_assertion_count
        for a in artifacts
    )

    ast_parse_failure_count = sum(
        int(
            a.ast_status
            in {
                "syntax_error",
                "parse_error",
                "read_error",
            }
        )
        for a in artifacts
    )

    ast_skipped_large_count = sum(
        int(
            a.ast_status
            == "skipped_large"
        )
        for a in artifacts
    )

    testcode_uses_pytest = any(
        a.imports_pytest
        for a in artifacts
    )

    testcode_uses_unittest = any(
        a.imports_unittest
        for a in artifacts
    )

    testcode_uses_doctest = any(
        a.imports_doctest
        for a in artifacts
    )

    uses_fixture = any(
        a.uses_fixture
        for a in artifacts
    )

    uses_parametrize = any(
        a.uses_parametrize
        for a in artifacts
    )

    uses_mock = any(
        a.uses_mock
        for a in artifacts
    )

    uses_monkeypatch = any(
        a.uses_monkeypatch
        for a in artifacts
    )

    uses_exception_testing = any(
        a.uses_exception_testing
        for a in artifacts
    )

    # --------------------------------------------------------
    # Three test definitions
    # --------------------------------------------------------

    has_broad = (
        test_artifact_file_count
        > 0
    )

    has_strict = any(
        a.has_strong_evidence
        for a in artifacts
    )

    has_standard_named = (
        standard_named_count
        > 0
    )

    has_conventional = (
        has_strict
        or has_standard_named
    )

    weak_category = (
        classify_weak_evidence(
            artifacts,
            has_strict=(
                has_strict
            ),
        )
    )

    test_artifact_loc_ratio = None

    if python_loc > 0:

        test_artifact_loc_ratio = (
            test_artifact_loc
            / python_loc
        )

    has_ci = (
        has_github_actions
        or has_travis_ci
        or has_circle_ci
        or has_gitlab_ci
    )

    if (
        declares_pytest_cov
        or declares_coverage
    ):
        has_coverage_tooling = True

    elapsed = (
        time.time()
        - started
    )

    result = RepositoryResult(
        repo_id=repo_id,
        cohort=cohort,
        full_name=full_name,
        clone_path=str(
            repo_path
        ),

        analysis_status="ok",
        analysis_elapsed_seconds=elapsed,
        error_message=None,

        python_file_count=(
            python_file_count
        ),

        python_loc=(
            python_loc
        ),

        production_python_file_count=(
            production_python_file_count
        ),

        production_python_loc=(
            production_python_loc
        ),

        test_artifact_file_count=(
            test_artifact_file_count
        ),

        test_artifact_loc=(
            test_artifact_loc
        ),

        test_artifact_loc_ratio=(
            test_artifact_loc_ratio
        ),

        standard_named_test_file_count=(
            standard_named_count
        ),

        test_directory_only_file_count=(
            test_directory_only_count
        ),

        init_test_file_count=(
            init_count
        ),

        has_tests_broad=int(
            has_broad
        ),

        has_tests_conventional=int(
            has_conventional
        ),

        has_tests_strict=int(
            has_strict
        ),

        weak_evidence_category=(
            weak_category
        ),

        test_function_count=(
            test_function_count
        ),

        test_class_count=(
            test_class_count
        ),

        assert_statement_count=(
            assert_statement_count
        ),

        unittest_assertion_count=(
            unittest_assertion_count
        ),

        ast_parse_failure_count=(
            ast_parse_failure_count
        ),

        ast_skipped_large_count=(
            ast_skipped_large_count
        ),

        testcode_uses_pytest=int(
            testcode_uses_pytest
        ),

        testcode_uses_unittest=int(
            testcode_uses_unittest
        ),

        testcode_uses_doctest=int(
            testcode_uses_doctest
        ),

        uses_fixture=int(
            uses_fixture
        ),

        uses_parametrize=int(
            uses_parametrize
        ),

        uses_mock=int(
            uses_mock
        ),

        uses_monkeypatch=int(
            uses_monkeypatch
        ),

        uses_exception_testing=int(
            uses_exception_testing
        ),

        has_conftest=int(
            has_conftest
        ),

        has_pytest_ini=int(
            has_pytest_ini
        ),

        has_tox_ini=int(
            has_tox_ini
        ),

        has_noxfile=int(
            has_noxfile
        ),

        has_pyproject=int(
            has_pyproject
        ),

        has_setup_cfg=int(
            has_setup_cfg
        ),

        has_setup_py=int(
            has_setup_py
        ),

        has_requirements=int(
            has_requirements
        ),

        has_pipfile=int(
            has_pipfile
        ),

        declares_pytest=int(
            declares_pytest
        ),

        declares_pytest_cov=int(
            declares_pytest_cov
        ),

        declares_coverage=int(
            declares_coverage
        ),

        declares_tox=int(
            declares_tox
        ),

        declares_nox=int(
            declares_nox
        ),

        declares_hypothesis=int(
            declares_hypothesis
        ),

        has_ci=int(
            has_ci
        ),

        has_github_actions=int(
            has_github_actions
        ),

        has_travis_ci=int(
            has_travis_ci
        ),

        has_circle_ci=int(
            has_circle_ci
        ),

        has_gitlab_ci=int(
            has_gitlab_ci
        ),

        has_ci_test_execution=int(
            has_ci_test_execution
        ),

        ci_uses_pytest=int(
            ci_uses_pytest
        ),

        ci_uses_unittest=int(
            ci_uses_unittest
        ),

        ci_uses_tox=int(
            ci_uses_tox
        ),

        ci_uses_nox=int(
            ci_uses_nox
        ),

        has_coverage_config=int(
            has_coverage_config
        ),

        has_coverage_tooling=int(
            has_coverage_tooling
        ),
    )

    return (
        result,
        artifacts,
    )


# ============================================================
# Empty / failed repository result
# ============================================================

def empty_repository_result(
    repo_id: int,
    cohort: str,
    full_name: str,
    clone_path: str,
    status: str,
    error_message: str,
    elapsed: float,
) -> RepositoryResult:

    return RepositoryResult(
        repo_id=repo_id,
        cohort=cohort,
        full_name=full_name,
        clone_path=clone_path,

        analysis_status=status,
        analysis_elapsed_seconds=elapsed,
        error_message=error_message,

        python_file_count=0,
        python_loc=0,

        production_python_file_count=0,
        production_python_loc=0,

        test_artifact_file_count=0,
        test_artifact_loc=0,
        test_artifact_loc_ratio=None,

        standard_named_test_file_count=0,
        test_directory_only_file_count=0,
        init_test_file_count=0,

        has_tests_broad=0,
        has_tests_conventional=0,
        has_tests_strict=0,

        weak_evidence_category="none",

        test_function_count=0,
        test_class_count=0,

        assert_statement_count=0,
        unittest_assertion_count=0,

        ast_parse_failure_count=0,
        ast_skipped_large_count=0,

        testcode_uses_pytest=0,
        testcode_uses_unittest=0,
        testcode_uses_doctest=0,

        uses_fixture=0,
        uses_parametrize=0,
        uses_mock=0,
        uses_monkeypatch=0,
        uses_exception_testing=0,

        has_conftest=0,
        has_pytest_ini=0,
        has_tox_ini=0,
        has_noxfile=0,

        has_pyproject=0,
        has_setup_cfg=0,
        has_setup_py=0,
        has_requirements=0,
        has_pipfile=0,

        declares_pytest=0,
        declares_pytest_cov=0,
        declares_coverage=0,
        declares_tox=0,
        declares_nox=0,
        declares_hypothesis=0,

        has_ci=0,

        has_github_actions=0,
        has_travis_ci=0,
        has_circle_ci=0,
        has_gitlab_ci=0,

        has_ci_test_execution=0,

        ci_uses_pytest=0,
        ci_uses_unittest=0,
        ci_uses_tox=0,
        ci_uses_nox=0,

        has_coverage_config=0,
        has_coverage_tooling=0,
    )


# ============================================================
# Save results
# ============================================================

def save_result(
    conn: sqlite3.Connection,
    result: RepositoryResult,
    artifacts: list[
        TestArtifactResult
    ],
):

    values = asdict(result)

    columns = list(
        values.keys()
    )

    placeholders = ", ".join(
        ["?"] * len(columns)
    )

    conn.execute(
        f"""
        INSERT OR REPLACE INTO repository_metrics (
            {", ".join(columns)},
            processed_at
        )
        VALUES (
            {placeholders},
            CURRENT_TIMESTAMP
        )
        """,
        tuple(
            values[c]
            for c in columns
        ),
    )

    conn.execute(
        """
        DELETE FROM test_artifacts
        WHERE repo_id = ?
        """,
        (
            result.repo_id,
        ),
    )

    for artifact in artifacts:

        values = asdict(
            artifact
        )

        columns = list(
            values.keys()
        )

        placeholders = ", ".join(
            ["?"] * len(columns)
        )

        conn.execute(
            f"""
            INSERT OR REPLACE INTO test_artifacts (
                {", ".join(columns)}
            )
            VALUES (
                {placeholders}
            )
            """,
            tuple(
                values[c]
                for c in columns
            ),
        )

    conn.commit()


# ============================================================
# Existing state
# ============================================================

def processed_repo_ids(
    conn: sqlite3.Connection,
) -> set[int]:

    rows = conn.execute(
        """
        SELECT repo_id
        FROM repository_metrics
        """
    ).fetchall()

    return {
        int(row[0])
        for row in rows
    }


# ============================================================
# Export
# ============================================================

def export_results(
    conn: sqlite3.Connection,
    config: dict,
):

    output = config[
        "output"
    ]

    repository_path = Path(
        output[
            "repository_metrics"
        ]
    )

    artifact_path = Path(
        output[
            "test_artifacts"
        ]
    )

    repository_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    repository_df = (
        pd.read_sql_query(
            """
            SELECT *
            FROM repository_metrics
            ORDER BY
                cohort,
                repo_id
            """,
            conn,
        )
    )

    repository_df.to_csv(
        repository_path,
        index=False,
    )

    artifact_df = (
        pd.read_sql_query(
            """
            SELECT
                a.*,
                r.cohort,
                r.full_name
            FROM test_artifacts a
            JOIN repository_metrics r
              ON a.repo_id = r.repo_id
            ORDER BY
                r.cohort,
                a.repo_id,
                a.path
            """,
            conn,
        )
    )

    artifact_df.to_csv(
        artifact_path,
        index=False,
    )

    print()
    print(
        f"Saved: "
        f"{repository_path}"
    )

    print(
        f"Saved: "
        f"{artifact_path}"
    )


# ============================================================
# Summary
# ============================================================

def print_statistics(
    conn: sqlite3.Connection,
):

    df = pd.read_sql_query(
        """
        SELECT *
        FROM repository_metrics
        """,
        conn,
    )

    if df.empty:
        return

    print()
    print(
        "=" * 70
    )

    print(
        "ANALYSIS STATISTICS"
    )

    print(
        "=" * 70
    )

    for cohort in COHORTS:

        g = df[
            df[
                "cohort"
            ] == cohort
        ]

        if g.empty:
            continue

        print()
        print(cohort)

        print(
            f"processed: "
            f"{len(g):,}"
        )

        print(
            "status:"
        )

        print(
            g[
                "analysis_status"
            ].value_counts()
        )

        ok = g[
            g[
                "analysis_status"
            ] == "ok"
        ]

        if ok.empty:
            continue

        print()

        for column, label in [
            (
                "has_tests_broad",
                "tests broad",
            ),
            (
                "has_tests_conventional",
                "tests conventional",
            ),
            (
                "has_tests_strict",
                "tests strict",
            ),
            (
                "has_ci",
                "has CI",
            ),
            (
                "has_ci_test_execution",
                "CI test execution",
            ),
            (
                "testcode_uses_pytest",
                "pytest in test code",
            ),
            (
                "testcode_uses_unittest",
                "unittest in test code",
            ),
            (
                "has_coverage_tooling",
                "coverage tooling",
            ),
        ]:

            count = int(
                ok[column].sum()
            )

            percentage = (
                ok[column].mean()
                * 100
            )

            print(
                f"{label}: "
                f"{count:,} "
                f"({percentage:.2f}%)"
            )

        print()

        print(
            "weak evidence:"
        )

        weak = ok[
            ok[
                "has_tests_broad"
            ] == 1
        ]

        weak = weak[
            weak[
                "has_tests_strict"
            ] == 0
        ]

        if not weak.empty:

            print(
                weak[
                    "weak_evidence_category"
                ]
                .value_counts()
                .to_string()
            )

        failed_repos = (
            ok[
                "ast_parse_failure_count"
            ] > 0
        ).sum()

        print()

        print(
            "repositories with "
            "AST failures: "
            f"{failed_repos:,} "
            f"({failed_repos / len(ok) * 100:.2f}%)"
        )

        times = (
            ok[
                "analysis_elapsed_seconds"
            ]
            .dropna()
        )

        if not times.empty:

            print(
                f"time median: "
                f"{times.median():.3f}s"
            )

            print(
                f"time mean:   "
                f"{times.mean():.3f}s"
            )

            print(
                f"time p95:    "
                f"{times.quantile(.95):.3f}s"
            )


# ============================================================
# Reset
# ============================================================

def reset_outputs(
    config: dict,
):

    output = config[
        "output"
    ]

    paths = [
        Path(
            output[
                "analysis_state"
            ]
        ),

        Path(
            output[
                "repository_metrics"
            ]
        ),

        Path(
            output[
                "test_artifacts"
            ]
        ),
    ]

    # SQLite WAL files
    db = paths[0]

    paths.extend([
        Path(
            str(db)
            + "-wal"
        ),

        Path(
            str(db)
            + "-shm"
        ),
    ])

    for path in paths:

        if path.exists():

            path.unlink()

            print(
                f"Deleted: {path}"
            )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pilot",
        type=int,
        default=None,
        help=(
            "Analyze at most N new "
            "repositories per cohort."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete all previous 04 "
            "state and outputs before running."
        ),
    )

    args = parser.parse_args()

    config = load_config()

    if args.reset:

        reset_outputs(
            config
        )

    output = config[
        "output"
    ]

    analysis_config = config[
        "analysis"
    ]

    selected_path = Path(
        output[
            "selected_repositories"
        ]
    )

    state_path = Path(
        output[
            "analysis_state"
        ]
    )

    workers = int(
        args.workers
        if args.workers is not None
        else analysis_config.get(
            "workers",
            4,
        )
    )

    selected = pd.read_csv(
        selected_path
    )

    selected = (
        selected
        .sort_values([
            "cohort",
            "selection_rank",
        ])
        .reset_index(
            drop=True
        )
    )

    conn = connect_database(
        state_path
    )

    already_processed = (
        processed_repo_ids(
            conn
        )
    )

    started = time.time()

    print()
    print(
        "=" * 70
    )

    print(
        "TEST INFRASTRUCTURE ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        f"workers: "
        f"{workers}"
    )

    print(
        "selected repositories: "
        f"{len(selected):,}"
    )

    interrupted = False

    try:

        for cohort in COHORTS:

            cohort_df = selected[
                selected[
                    "cohort"
                ] == cohort
            ]

            pending = [
                row.to_dict()
                for _, row
                in cohort_df.iterrows()
                if int(
                    row[
                        "repo_id"
                    ]
                )
                not in already_processed
            ]

            if args.pilot is not None:

                pending = (
                    pending[
                        :args.pilot
                    ]
                )

            print()
            print(
                "=" * 70
            )

            print(
                f"{cohort}: "
                f"{len(pending):,} "
                "repositories to analyze"
            )

            print(
                "=" * 70
            )

            processed_this_run = 0

            batch_size = max(
                workers * 8,
                workers,
            )

            for start in range(
                0,
                len(pending),
                batch_size,
            ):

                batch = pending[
                    start:
                    start
                    + batch_size
                ]

                results = {}

                with ThreadPoolExecutor(
                    max_workers=workers
                ) as executor:

                    future_map = {}

                    for position, row in enumerate(
                        batch
                    ):

                        future = (
                            executor.submit(
                                analyze_repository,
                                row,
                                config,
                            )
                        )

                        future_map[
                            future
                        ] = (
                            position,
                            row,
                        )

                    for future in as_completed(
                        future_map
                    ):

                        (
                            position,
                            row,
                        ) = future_map[
                            future
                        ]

                        try:

                            results[
                                position
                            ] = future.result()

                        except Exception as e:

                            result = (
                                empty_repository_result(
                                    repo_id=int(
                                        row[
                                            "repo_id"
                                        ]
                                    ),
                                    cohort=str(
                                        row[
                                            "cohort"
                                        ]
                                    ),
                                    full_name=str(
                                        row[
                                            "full_name"
                                        ]
                                    ),
                                    clone_path=str(
                                        row[
                                            "clone_path"
                                        ]
                                    ),
                                    status=(
                                        "analysis_failed"
                                    ),
                                    error_message=(
                                        f"{type(e).__name__}: "
                                        f"{e}"
                                    ),
                                    elapsed=0.0,
                                )
                            )

                            results[
                                position
                            ] = (
                                result,
                                [],
                            )

                # Deterministic write order.
                for position, row in enumerate(
                    batch
                ):

                    (
                        result,
                        artifacts,
                    ) = results[
                        position
                    ]

                    save_result(
                        conn,
                        result,
                        artifacts,
                    )

                    already_processed.add(
                        result.repo_id
                    )

                    processed_this_run += 1

                    print(
                        f"[{cohort}] "
                        f"processed="
                        f"{processed_this_run:,}/"
                        f"{len(pending):,} "
                        f"repo="
                        f"{result.full_name} "
                        f"py="
                        f"{result.python_file_count:,} "
                        f"broad="
                        f"{result.has_tests_broad} "
                        f"conv="
                        f"{result.has_tests_conventional} "
                        f"strict="
                        f"{result.has_tests_strict} "
                        f"time="
                        f"{result.analysis_elapsed_seconds:.3f}s"
                    )

                if (
                    processed_this_run
                    % 2000
                    == 0
                ):

                    export_results(
                        conn,
                        config,
                    )

    except KeyboardInterrupt:

        interrupted = True

        print()
        print(
            "Interrupted by user."
        )

        print(
            "Completed repository analyses "
            "remain in SQLite."
        )

    finally:

        export_results(
            conn,
            config,
        )

        print_statistics(
            conn
        )

        elapsed = (
            time.time()
            - started
        )

        print()
        print(
            "=" * 70
        )

        if interrupted:
            print(
                "ANALYSIS STOPPED"
            )
        else:
            print(
                "ANALYSIS FINISHED"
            )

        print(
            "=" * 70
        )

        print(
            f"Elapsed: "
            f"{elapsed / 3600:.2f} hours"
        )

        conn.close()


if __name__ == "__main__":
    main()
