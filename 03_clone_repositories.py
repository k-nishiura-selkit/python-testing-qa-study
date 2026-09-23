from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
import time

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
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

    with CONFIG_PATH.open(
        encoding="utf-8"
    ) as f:

        return yaml.safe_load(f)


# ============================================================
# Result objects
# ============================================================

@dataclass
class AcquisitionResult:

    repo_id: int
    cohort: str
    full_name: str

    status: str

    elapsed_seconds: float

    error_message: str | None

    temp_path: str | None

    acquisition_method: str | None

    archive_branch: str | None


@dataclass
class FinalCloneResult:

    repo_id: int
    cohort: str
    full_name: str

    status: str

    elapsed_seconds: float

    clone_size_bytes: int | None

    python_file_count: int | None
    total_file_count: int | None

    error_message: str | None

    temp_path: str | None

    acquisition_method: str | None

    archive_branch: str | None


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

    conn = sqlite3.connect(
        path
    )

    conn.execute(
        "PRAGMA journal_mode=WAL"
    )

    conn.execute(
        "PRAGMA synchronous=NORMAL"
    )

    create_tables(
        conn
    )

    ensure_schema(
        conn
    )

    return conn


def create_tables(
    conn: sqlite3.Connection,
):

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS clone_results (

            repo_id INTEGER PRIMARY KEY,

            cohort TEXT NOT NULL,
            random_key TEXT NOT NULL,
            full_name TEXT NOT NULL,

            clone_status TEXT NOT NULL,

            clone_path TEXT,

            clone_elapsed_seconds REAL,
            clone_size_bytes INTEGER,

            python_file_count INTEGER,
            total_file_count INTEGER,

            error_message TEXT,

            acquisition_method TEXT,
            archive_branch TEXT,

            final_selected INTEGER
                NOT NULL DEFAULT 0,

            selection_rank INTEGER,

            processed_at TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_clone_cohort_status
        ON clone_results(
            cohort,
            clone_status
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_clone_selection
        ON clone_results(
            cohort,
            final_selected,
            selection_rank
        )
        """
    )

    conn.commit()


def ensure_schema(
    conn: sqlite3.Connection,
):
    """
    Add columns to an existing 03 database created
    by an older version of the script.
    """

    existing = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(clone_results)"
        )
    }

    additions = {
        "acquisition_method":
            "TEXT",

        "archive_branch":
            "TEXT",
    }

    for column, column_type in additions.items():

        if column not in existing:

            conn.execute(
                f"""
                ALTER TABLE clone_results
                ADD COLUMN {column} {column_type}
                """
            )

    conn.commit()


# ============================================================
# Disk helpers
# ============================================================

def free_space_gb(
    path: Path,
) -> float:

    usage = shutil.disk_usage(
        path
    )

    return (
        usage.free
        / 1024
        / 1024
        / 1024
    )


def directory_size_bytes(
    path: Path,
) -> int:

    total = 0

    for root, dirs, files in os.walk(
        path
    ):

        for filename in files:

            file_path = (
                Path(root)
                / filename
            )

            try:

                total += (
                    file_path.stat().st_size
                )

            except (
                FileNotFoundError,
                PermissionError,
            ):

                pass

    return total


# ============================================================
# Repository inspection
# ============================================================

def inspect_repository(
    repo_path: Path,
    exclude_dirs: set[str],
) -> tuple[int, int]:

    python_files = 0
    total_files = 0

    for root, dirs, files in os.walk(
        repo_path
    ):

        dirs[:] = [
            dirname
            for dirname in dirs
            if dirname not in exclude_dirs
        ]

        for filename in files:

            total_files += 1

            if filename.lower().endswith(
                ".py"
            ):

                python_files += 1

    return (
        python_files,
        total_files,
    )


# ============================================================
# Anonymous GitHub public-repository check
# ============================================================

def check_public_repository(
    full_name: str,
) -> tuple[
    str,
    str | None,
]:
    """
    Uses NO GitHub authentication.

    Returns:
        (status, default_branch)

    status:
        public
        not_found
        rate_limited
        server_error
        network_error
        unknown
    """

    try:

        response = requests.get(
            (
                "https://api.github.com/repos/"
                + full_name
            ),
            headers={
                "Accept":
                    "application/vnd.github+json",

                "X-GitHub-Api-Version":
                    "2022-11-28",

                "User-Agent":
                    "python-testing-foundation-study",
            },
            timeout=30,
        )

    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
    ):

        return (
            "network_error",
            None,
        )

    if response.status_code == 200:

        try:

            data = response.json()

            default_branch = (
                data.get(
                    "default_branch"
                )
            )

        except Exception:

            default_branch = None

        return (
            "public",
            default_branch,
        )

    if response.status_code == 404:

        return (
            "not_found",
            None,
        )

    if response.status_code in {
        403,
        429,
    }:

        return (
            "rate_limited",
            None,
        )

    if (
        500
        <= response.status_code
        < 600
    ):

        return (
            "server_error",
            None,
        )

    return (
        "unknown",
        None,
    )


# ============================================================
# Error classification
# ============================================================

def has_transient_signature(
    text: str | None,
) -> bool:

    if not text:
        return False

    text = text.lower()

    signatures = [
        "expected flush after ref listing",
        "connection reset by peer",
        "connection timed out",
        "operation timed out",
        "failed to connect to github.com",
        "couldn't connect to server",
        "could not resolve host",
        "remote end hung up unexpectedly",
        "the remote end hung up unexpectedly",
        "early eof",
        "429",
        "500",
        "502",
        "503",
        "504",
    ]

    return any(
        signature in text
        for signature in signatures
    )


# ============================================================
# Git shallow clone
# ============================================================

def try_git_clone(
    row: dict,
    temp_path: Path,
    timeout_seconds: int,
) -> tuple[
    bool,
    float,
    str | None,
]:

    clone_url = str(
        row[
            "clone_url"
        ]
    )

    if temp_path.exists():

        shutil.rmtree(
            temp_path,
            ignore_errors=True,
        )

    temp_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        "git",

        # Do not use local credentials.
        "-c",
        "credential.helper=",

        "clone",

        "--depth",
        "1",

        "--single-branch",

        "--no-tags",

        "--quiet",

        clone_url,

        str(
            temp_path
        ),
    ]

    env = os.environ.copy()

    # Never block waiting for:
    #
    # Username for 'https://github.com':
    env[
        "GIT_TERMINAL_PROMPT"
    ] = "0"

    started = time.time()

    try:

        completed = subprocess.run(
            command,

            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,

            text=True,

            timeout=timeout_seconds,

            env=env,

            stdin=subprocess.DEVNULL,
        )

    except subprocess.TimeoutExpired:

        elapsed = (
            time.time()
            - started
        )

        shutil.rmtree(
            temp_path,
            ignore_errors=True,
        )

        return (
            False,
            elapsed,
            (
                f"git clone timeout after "
                f"{timeout_seconds}s"
            ),
        )

    except Exception as e:

        elapsed = (
            time.time()
            - started
        )

        shutil.rmtree(
            temp_path,
            ignore_errors=True,
        )

        return (
            False,
            elapsed,
            (
                f"{type(e).__name__}: "
                f"{e}"
            ),
        )

    elapsed = (
        time.time()
        - started
    )

    if completed.returncode == 0:

        return (
            True,
            elapsed,
            None,
        )

    error = (
        completed.stderr
        or ""
    ).strip()

    shutil.rmtree(
        temp_path,
        ignore_errors=True,
    )

    return (
        False,
        elapsed,
        error,
    )


# ============================================================
# Safe tar extraction
# ============================================================

def safe_extract_tar(
    archive_path: Path,
    extraction_dir: Path,
):

    extraction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base = extraction_dir.resolve()

    with tarfile.open(
        archive_path,
        "r:gz",
    ) as tar:

        for member in tar.getmembers():

            target = (
                extraction_dir
                / member.name
            ).resolve()

            if (
                target != base
                and base not in target.parents
            ):

                raise RuntimeError(
                    "Unsafe path detected "
                    "in source archive."
                )

        tar.extractall(
            extraction_dir
        )


# ============================================================
# Archive download fallback
# ============================================================

def try_archive_download(
    full_name: str,
    branch: str,
    temp_path: Path,
    timeout_seconds: int,
) -> tuple[
    bool,
    float,
    str | None,
]:
    """
    Download the current default branch as a tar.gz snapshot.

    Uses GitHub's public codeload endpoint and no credentials.
    """

    if temp_path.exists():

        shutil.rmtree(
            temp_path,
            ignore_errors=True,
        )

    temp_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # codeload handles the raw source archive.
    url = (
        "https://codeload.github.com/"
        f"{full_name}/tar.gz/refs/heads/"
        f"{branch}"
    )

    started = time.time()

    archive_file = None
    extraction_dir = None

    try:

        with tempfile.NamedTemporaryFile(
            prefix="github-archive-",
            suffix=".tar.gz",
            delete=False,
        ) as tmp:

            archive_file = Path(
                tmp.name
            )

            with requests.get(
                url,
                stream=True,
                timeout=timeout_seconds,
                headers={
                    "User-Agent":
                        "python-testing-foundation-study",
                },
            ) as response:

                if response.status_code != 200:

                    elapsed = (
                        time.time()
                        - started
                    )

                    return (
                        False,
                        elapsed,
                        (
                            "archive HTTP "
                            f"{response.status_code}"
                        ),
                    )

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:

                        tmp.write(
                            chunk
                        )

        extraction_dir = Path(
            tempfile.mkdtemp(
                prefix="github-extract-"
            )
        )

        safe_extract_tar(
            archive_file,
            extraction_dir,
        )

        # GitHub source archives normally contain
        # exactly one top-level directory:
        #
        #   repo-<commit>/
        #
        children = list(
            extraction_dir.iterdir()
        )

        if len(children) != 1:

            raise RuntimeError(
                "Unexpected archive structure: "
                f"{len(children)} top-level items"
            )

        source_root = children[0]

        if not source_root.is_dir():

            raise RuntimeError(
                "Archive top-level item "
                "is not a directory."
            )

        # Move the extracted source tree into
        # the same temporary repository location
        # used by successful Git clones.
        shutil.move(
            str(source_root),
            str(temp_path),
        )

        elapsed = (
            time.time()
            - started
        )

        return (
            True,
            elapsed,
            None,
        )

    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
    ) as e:

        elapsed = (
            time.time()
            - started
        )

        shutil.rmtree(
            temp_path,
            ignore_errors=True,
        )

        return (
            False,
            elapsed,
            (
                f"{type(e).__name__}: "
                f"{e}"
            ),
        )

    except Exception as e:

        elapsed = (
            time.time()
            - started
        )

        shutil.rmtree(
            temp_path,
            ignore_errors=True,
        )

        return (
            False,
            elapsed,
            (
                f"{type(e).__name__}: "
                f"{e}"
            ),
        )

    finally:

        if (
            archive_file is not None
            and archive_file.exists()
        ):

            try:
                archive_file.unlink()
            except Exception:
                pass

        if (
            extraction_dir is not None
            and extraction_dir.exists()
        ):

            shutil.rmtree(
                extraction_dir,
                ignore_errors=True,
            )


# ============================================================
# Acquire one repository
# ============================================================

def acquire_repository(
    row: dict,
    temp_root: Path,
    git_timeout_seconds: int,
    archive_timeout_seconds: int,
) -> AcquisitionResult:

    repo_id = int(
        row[
            "repo_id"
        ]
    )

    cohort = str(
        row[
            "cohort"
        ]
    )

    full_name = str(
        row[
            "full_name"
        ]
    )

    temp_path = (
        temp_root
        / cohort
        / str(
            repo_id
        )
    )

    total_started = (
        time.time()
    )

    # --------------------------------------------------------
    # 1. First choice: shallow Git clone
    # --------------------------------------------------------

    (
        git_success,
        git_elapsed,
        git_error,
    ) = try_git_clone(
        row=row,
        temp_path=temp_path,
        timeout_seconds=(
            git_timeout_seconds
        ),
    )

    if git_success:

        return AcquisitionResult(
            repo_id=repo_id,
            cohort=cohort,
            full_name=full_name,

            status="success",

            elapsed_seconds=(
                time.time()
                - total_started
            ),

            error_message=None,

            temp_path=str(
                temp_path
            ),

            acquisition_method="git",

            archive_branch=None,
        )

    # --------------------------------------------------------
    # 2. Git failed:
    #    check anonymous public visibility
    # --------------------------------------------------------

    (
        public_status,
        current_default_branch,
    ) = check_public_repository(
        full_name
    )

    # Repository deleted or now private.
    if public_status == "not_found":

        return AcquisitionResult(
            repo_id=repo_id,
            cohort=cohort,
            full_name=full_name,

            status="permanent_not_found",

            elapsed_seconds=(
                time.time()
                - total_started
            ),

            error_message=(
                git_error
            ),

            temp_path=None,

            acquisition_method=None,

            archive_branch=None,
        )

    # Could not reliably check GitHub status.
    # Do not reject the repository.
    if public_status != "public":

        return AcquisitionResult(
            repo_id=repo_id,
            cohort=cohort,
            full_name=full_name,

            status="transient_failure",

            elapsed_seconds=(
                time.time()
                - total_started
            ),

            error_message=(
                "Git clone failed:\n"
                f"{git_error}\n\n"
                "Anonymous repository check: "
                f"{public_status}"
            ),

            temp_path=None,

            acquisition_method=None,

            archive_branch=None,
        )

    # --------------------------------------------------------
    # 3. Public repository:
    #    use source archive fallback.
    # --------------------------------------------------------

    branch = (
        current_default_branch
    )

    if not branch:

        # Fallback to 02 metadata only if the
        # current REST response did not provide one.
        candidate_branch = row.get(
            "default_branch"
        )

        if (
            candidate_branch is not None
            and pd.notna(
                candidate_branch
            )
            and str(
                candidate_branch
            ).strip()
        ):

            branch = str(
                candidate_branch
            )

    if not branch:

        return AcquisitionResult(
            repo_id=repo_id,
            cohort=cohort,
            full_name=full_name,

            status="transient_failure",

            elapsed_seconds=(
                time.time()
                - total_started
            ),

            error_message=(
                "Git clone failed and "
                "default branch could not "
                "be determined.\n"
                f"{git_error}"
            ),

            temp_path=None,

            acquisition_method=None,

            archive_branch=None,
        )

    print(
        f"  -> archive fallback: "
        f"{full_name} "
        f"(branch={branch})"
    )

    (
        archive_success,
        archive_elapsed,
        archive_error,
    ) = try_archive_download(
        full_name=full_name,
        branch=branch,
        temp_path=temp_path,
        timeout_seconds=(
            archive_timeout_seconds
        ),
    )

    if archive_success:

        return AcquisitionResult(
            repo_id=repo_id,
            cohort=cohort,
            full_name=full_name,

            status="success",

            elapsed_seconds=(
                time.time()
                - total_started
            ),

            error_message=(
                git_error
            ),

            temp_path=str(
                temp_path
            ),

            acquisition_method="archive",

            archive_branch=branch,
        )

    # --------------------------------------------------------
    # Both Git and archive failed.
    #
    # Public repository is known to exist, so we avoid
    # excluding it immediately.
    # --------------------------------------------------------

    combined_error = (
        "Git clone failed:\n"
        f"{git_error}\n\n"
        "Archive fallback failed:\n"
        f"{archive_error}"
    )

    # HTTP 404 from codeload despite a public REST record
    # can occur during GitHub internal transitions.
    # Treat conservatively as transient.
    return AcquisitionResult(
        repo_id=repo_id,
        cohort=cohort,
        full_name=full_name,

        status="transient_failure",

        elapsed_seconds=(
            time.time()
            - total_started
        ),

        error_message=(
            combined_error
        ),

        temp_path=None,

        acquisition_method=None,

        archive_branch=branch,
    )


# ============================================================
# Inspect acquired repository
# ============================================================

def finalize_acquisition(
    row: dict,
    acquisition: AcquisitionResult,
    exclude_dirs: set[str],
) -> FinalCloneResult:

    repo_id = int(
        row[
            "repo_id"
        ]
    )

    cohort = str(
        row[
            "cohort"
        ]
    )

    full_name = str(
        row[
            "full_name"
        ]
    )

    temp_path = Path(
        acquisition.temp_path
    )

    try:

        (
            python_file_count,
            total_file_count,
        ) = inspect_repository(
            temp_path,
            exclude_dirs,
        )

        clone_size = (
            directory_size_bytes(
                temp_path
            )
        )

    except Exception as e:

        shutil.rmtree(
            temp_path,
            ignore_errors=True,
        )

        return FinalCloneResult(
            repo_id=repo_id,
            cohort=cohort,
            full_name=full_name,

            status="inspection_failed",

            elapsed_seconds=(
                acquisition.elapsed_seconds
            ),

            clone_size_bytes=None,

            python_file_count=None,
            total_file_count=None,

            error_message=(
                f"{type(e).__name__}: "
                f"{e}"
            ),

            temp_path=None,

            acquisition_method=(
                acquisition.acquisition_method
            ),

            archive_branch=(
                acquisition.archive_branch
            ),
        )

    # --------------------------------------------------------
    # No first-party Python source
    # --------------------------------------------------------

    if python_file_count == 0:

        shutil.rmtree(
            temp_path,
            ignore_errors=True,
        )

        return FinalCloneResult(
            repo_id=repo_id,
            cohort=cohort,
            full_name=full_name,

            status="no_python_source",

            elapsed_seconds=(
                acquisition.elapsed_seconds
            ),

            clone_size_bytes=(
                clone_size
            ),

            python_file_count=0,

            total_file_count=(
                total_file_count
            ),

            error_message=None,

            temp_path=None,

            acquisition_method=(
                acquisition.acquisition_method
            ),

            archive_branch=(
                acquisition.archive_branch
            ),
        )

    return FinalCloneResult(
        repo_id=repo_id,
        cohort=cohort,
        full_name=full_name,

        status="eligible",

        elapsed_seconds=(
            acquisition.elapsed_seconds
        ),

        clone_size_bytes=(
            clone_size
        ),

        python_file_count=(
            python_file_count
        ),

        total_file_count=(
            total_file_count
        ),

        error_message=(
            acquisition.error_message
        ),

        temp_path=str(
            temp_path
        ),

        acquisition_method=(
            acquisition.acquisition_method
        ),

        archive_branch=(
            acquisition.archive_branch
        ),
    )


# ============================================================
# Concurrent batch
# ============================================================

def run_acquisition_batch(
    rows: list[dict],
    temp_root: Path,
    git_timeout_seconds: int,
    archive_timeout_seconds: int,
    workers: int,
) -> dict[
    int,
    AcquisitionResult,
]:

    results = {}

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        future_map = {}

        for position, row in enumerate(
            rows
        ):

            future = executor.submit(
                acquire_repository,
                row,
                temp_root,
                git_timeout_seconds,
                archive_timeout_seconds,
            )

            future_map[
                future
            ] = position

        for future in as_completed(
            future_map
        ):

            position = (
                future_map[
                    future
                ]
            )

            row = rows[
                position
            ]

            try:

                results[
                    position
                ] = future.result()

            except Exception as e:

                results[
                    position
                ] = AcquisitionResult(
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

                    status=(
                        "transient_failure"
                    ),

                    elapsed_seconds=0.0,

                    error_message=(
                        f"{type(e).__name__}: "
                        f"{e}"
                    ),

                    temp_path=None,

                    acquisition_method=None,

                    archive_branch=None,
                )

    return results


# ============================================================
# Global cooldown / retry
# ============================================================

def resolve_batch_with_cooldown(
    batch: list[dict],
    temp_root: Path,
    git_timeout_seconds: int,
    archive_timeout_seconds: int,
    workers: int,
    cooldown_seconds: list[int],
) -> tuple[
    dict[int, AcquisitionResult],
    bool,
]:

    results = run_acquisition_batch(
        rows=batch,
        temp_root=temp_root,
        git_timeout_seconds=(
            git_timeout_seconds
        ),
        archive_timeout_seconds=(
            archive_timeout_seconds
        ),
        workers=workers,
    )

    for cooldown_index, wait_seconds in enumerate(
        cooldown_seconds,
        start=1,
    ):

        transient_positions = [
            position
            for position, result
            in results.items()
            if result.status
            == "transient_failure"
        ]

        if not transient_positions:

            return (
                results,
                False,
            )

        print()
        print(
            "!" * 70
        )

        print(
            "TRANSIENT ACQUISITION FAILURE DETECTED"
        )

        print(
            f"affected repositories: "
            f"{len(transient_positions)}"
        )

        for position in transient_positions:

            result = results[
                position
            ]

            print()
            print(
                f"  repo: "
                f"{result.full_name}"
            )

            if result.error_message:

                print(
                    "  error:"
                )

                for line in (
                    result
                    .error_message
                    .splitlines()
                ):

                    print(
                        "    "
                        + line
                    )

        print()

        print(
            f"GLOBAL COOLDOWN "
            f"{cooldown_index}/"
            f"{len(cooldown_seconds)}: "
            f"{wait_seconds} seconds"
        )

        print(
            "No new repository requests "
            "will be started during this period."
        )

        print(
            "!" * 70
        )

        time.sleep(
            wait_seconds
        )

        retry_rows = [
            batch[
                position
            ]
            for position
            in transient_positions
        ]

        retry_results = (
            run_acquisition_batch(
                rows=retry_rows,
                temp_root=temp_root,
                git_timeout_seconds=(
                    git_timeout_seconds
                ),
                archive_timeout_seconds=(
                    archive_timeout_seconds
                ),
                workers=min(
                    workers,
                    len(
                        retry_rows
                    ),
                ),
            )
        )

        for (
            retry_position,
            original_position,
        ) in enumerate(
            transient_positions
        ):

            results[
                original_position
            ] = retry_results[
                retry_position
            ]

    # --------------------------------------------------------
    # Still unresolved after all cooldowns
    # --------------------------------------------------------

    remaining = [
        position
        for position, result
        in results.items()
        if result.status
        == "transient_failure"
    ]

    if remaining:

        print()
        print(
            "!" * 70
        )

        print(
            "SAFE STOP REQUIRED"
        )

        print(
            f"{len(remaining)} repositories "
            "still cannot be acquired."
        )

        for position in remaining:

            result = results[
                position
            ]

            print()
            print(
                f"  repo: "
                f"{result.full_name}"
            )

            if result.error_message:

                print(
                    "  error:"
                )

                for line in (
                    result
                    .error_message
                    .splitlines()
                ):

                    print(
                        "    "
                        + line
                    )

        print()
        print(
            "They will NOT be recorded as failures."
        )

        print(
            "Restart 03 later to retry "
            "the same candidates."
        )

        print(
            "!" * 70
        )

        # This entire batch remains uncommitted.
        for result in results.values():

            if result.temp_path:

                path = Path(
                    result.temp_path
                )

                if path.exists():

                    shutil.rmtree(
                        path,
                        ignore_errors=True,
                    )

        return (
            results,
            True,
        )

    return (
        results,
        False,
    )


# ============================================================
# Startup cleanup
# ============================================================

def clean_incomplete_temp_dirs(
    temp_root: Path,
):

    if not temp_root.exists():
        return

    print(
        "Cleaning incomplete temporary repositories..."
    )

    for cohort in COHORTS:

        cohort_tmp = (
            temp_root
            / cohort
        )

        if not cohort_tmp.exists():
            continue

        for path in cohort_tmp.iterdir():

            if path.is_dir():

                shutil.rmtree(
                    path,
                    ignore_errors=True,
                )


# ============================================================
# Reconcile DB and filesystem
# ============================================================

def reconcile_database(
    conn: sqlite3.Connection,
    repositories_root: Path,
):

    rows = conn.execute(
        """
        SELECT
            repo_id,
            cohort,
            final_selected
        FROM clone_results
        """
    ).fetchall()

    for (
        repo_id,
        cohort,
        final_selected,
    ) in rows:

        if final_selected == 1:

            expected_path = (
                repositories_root
                / cohort
                / str(
                    repo_id
                )
            )

            if not expected_path.exists():

                print(
                    "WARNING: selected repository "
                    f"{repo_id} is missing. "
                    "Resetting for reprocessing."
                )

                conn.execute(
                    """
                    DELETE FROM clone_results
                    WHERE repo_id = ?
                    """,
                    (
                        repo_id,
                    ),
                )

    conn.commit()


# ============================================================
# Existing state
# ============================================================

def selected_count(
    conn: sqlite3.Connection,
    cohort: str,
) -> int:

    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM clone_results
        WHERE cohort = ?
          AND final_selected = 1
        """,
        (
            cohort,
        ),
    ).fetchone()

    return int(
        row[0]
    )


def next_selection_rank(
    conn: sqlite3.Connection,
    cohort: str,
) -> int:

    row = conn.execute(
        """
        SELECT COALESCE(
            MAX(selection_rank),
            0
        )
        FROM clone_results
        WHERE cohort = ?
          AND final_selected = 1
        """,
        (
            cohort,
        ),
    ).fetchone()

    return (
        int(
            row[0]
        )
        + 1
    )


def processed_repo_ids(
    conn: sqlite3.Connection,
) -> set[int]:

    rows = conn.execute(
        """
        SELECT repo_id
        FROM clone_results
        """
    ).fetchall()

    return {
        int(
            row[0]
        )
        for row in rows
    }


# ============================================================
# Database writes
# ============================================================

def save_rejection(
    conn: sqlite3.Connection,
    row: dict,
    result: FinalCloneResult,
):

    conn.execute(
        """
        INSERT OR REPLACE INTO clone_results (

            repo_id,
            cohort,
            random_key,
            full_name,

            clone_status,
            clone_path,

            clone_elapsed_seconds,
            clone_size_bytes,

            python_file_count,
            total_file_count,

            error_message,

            acquisition_method,
            archive_branch,

            final_selected,
            selection_rank,

            processed_at
        )
        VALUES (
            ?, ?, ?, ?,
            ?, NULL,
            ?, ?,
            ?, ?,
            ?,
            ?, ?,
            0, NULL,
            CURRENT_TIMESTAMP
        )
        """,
        (
            int(
                row[
                    "repo_id"
                ]
            ),

            row[
                "cohort"
            ],

            row[
                "random_key"
            ],

            row[
                "full_name"
            ],

            result.status,

            result.elapsed_seconds,

            result.clone_size_bytes,

            result.python_file_count,

            result.total_file_count,

            result.error_message,

            result.acquisition_method,

            result.archive_branch,
        ),
    )

    conn.commit()


def save_acceptance(
    conn: sqlite3.Connection,
    row: dict,
    result: FinalCloneResult,
    final_path: Path,
    selection_rank: int,
):

    conn.execute(
        """
        INSERT OR REPLACE INTO clone_results (

            repo_id,
            cohort,
            random_key,
            full_name,

            clone_status,
            clone_path,

            clone_elapsed_seconds,
            clone_size_bytes,

            python_file_count,
            total_file_count,

            error_message,

            acquisition_method,
            archive_branch,

            final_selected,
            selection_rank,

            processed_at
        )
        VALUES (
            ?, ?, ?, ?,
            'accepted', ?,
            ?, ?,
            ?, ?,
            ?,
            ?, ?,
            1, ?,
            CURRENT_TIMESTAMP
        )
        """,
        (
            int(
                row[
                    "repo_id"
                ]
            ),

            row[
                "cohort"
            ],

            row[
                "random_key"
            ],

            row[
                "full_name"
            ],

            str(
                final_path
            ),

            result.elapsed_seconds,

            result.clone_size_bytes,

            result.python_file_count,

            result.total_file_count,

            result.error_message,

            result.acquisition_method,

            result.archive_branch,

            selection_rank,
        ),
    )

    conn.commit()


# ============================================================
# Failed acquisition -> final rejection
# ============================================================

def acquisition_to_final_failure(
    acquisition: AcquisitionResult,
) -> FinalCloneResult:

    if (
        acquisition.status
        == "permanent_not_found"
    ):

        status = (
            "clone_failed_not_found"
        )

    else:

        raise ValueError(
            "Cannot convert acquisition status: "
            f"{acquisition.status}"
        )

    return FinalCloneResult(
        repo_id=acquisition.repo_id,
        cohort=acquisition.cohort,
        full_name=acquisition.full_name,

        status=status,

        elapsed_seconds=(
            acquisition.elapsed_seconds
        ),

        clone_size_bytes=None,

        python_file_count=None,
        total_file_count=None,

        error_message=(
            acquisition.error_message
        ),

        temp_path=None,

        acquisition_method=None,

        archive_branch=None,
    )


# ============================================================
# Process one cohort
# ============================================================

def process_cohort(
    conn: sqlite3.Connection,
    candidates: pd.DataFrame,
    cohort: str,
    repositories_root: Path,
    temp_root: Path,
    exclude_dirs: set[str],
    workers: int,
    git_timeout_seconds: int,
    archive_timeout_seconds: int,
    min_free_gb: float,
    cooldown_seconds: list[int],
    target: int,
    pilot: int | None,
) -> bool:

    cohort_df = (
        candidates[
            candidates[
                "cohort"
            ] == cohort
        ]
        .sort_values(
            "random_key"
        )
        .reset_index(
            drop=True
        )
    )

    already_processed = (
        processed_repo_ids(
            conn
        )
    )

    selected = selected_count(
        conn,
        cohort,
    )

    rank = next_selection_rank(
        conn,
        cohort,
    )

    print()
    print(
        "=" * 70
    )

    print(
        f"{cohort}: "
        f"{selected:,}/"
        f"{target:,} already selected"
    )

    print(
        "=" * 70
    )

    if selected >= target:

        return True

    pending_rows = []

    for _, row in cohort_df.iterrows():

        repo_id = int(
            row[
                "repo_id"
            ]
        )

        if repo_id in already_processed:
            continue

        pending_rows.append(
            row.to_dict()
        )

    if pilot is not None:

        pending_rows = (
            pending_rows[
                :pilot
            ]
        )

    processed_this_run = 0
    index = 0

    while (
        index
        < len(
            pending_rows
        )
        and selected
        < target
    ):

        free_gb = free_space_gb(
            repositories_root
        )

        if free_gb < min_free_gb:

            print()
            print(
                "SAFE STOP: free disk space "
                f"is {free_gb:.1f} GB, "
                f"below configured minimum "
                f"{min_free_gb:.1f} GB."
            )

            return False

        batch = (
            pending_rows[
                index:
                index + workers
            ]
        )

        (
            acquisitions,
            safe_stop,
        ) = resolve_batch_with_cooldown(
            batch=batch,
            temp_root=temp_root,
            git_timeout_seconds=(
                git_timeout_seconds
            ),
            archive_timeout_seconds=(
                archive_timeout_seconds
            ),
            workers=workers,
            cooldown_seconds=(
                cooldown_seconds
            ),
        )

        if safe_stop:

            return False

        index += len(
            batch
        )

        # ----------------------------------------------------
        # Evaluate strictly in random_key order
        # ----------------------------------------------------

        for position, row in enumerate(
            batch
        ):

            acquisition = acquisitions[
                position
            ]

            processed_this_run += 1

            print(
                f"[{cohort}] "
                f"processed="
                f"{processed_this_run:,} "
                f"selected="
                f"{selected:,}/"
                f"{target:,} "
                f"repo="
                f"{row['full_name']}"
            )

            # ------------------------------------------------
            # Permanently unavailable
            # ------------------------------------------------

            if (
                acquisition.status
                == "permanent_not_found"
            ):

                final_result = (
                    acquisition_to_final_failure(
                        acquisition
                    )
                )

                save_rejection(
                    conn,
                    row,
                    final_result,
                )

                print(
                    "  -> REJECT "
                    "clone_failed_not_found"
                )

                continue

            if (
                acquisition.status
                != "success"
            ):

                raise RuntimeError(
                    "Unexpected unresolved "
                    "acquisition state: "
                    f"{acquisition.status}"
                )

            # ------------------------------------------------
            # Successful Git clone OR archive fallback
            # ------------------------------------------------

            final_result = (
                finalize_acquisition(
                    row=row,
                    acquisition=acquisition,
                    exclude_dirs=(
                        exclude_dirs
                    ),
                )
            )

            free_gb = free_space_gb(
                repositories_root
            )

            if (
                final_result.clone_size_bytes
                is not None
            ):

                size_mb = (
                    final_result.clone_size_bytes
                    / 1024
                    / 1024
                )

                size_text = (
                    f"{size_mb:.1f} MB"
                )

            else:

                size_text = "-"

            # ------------------------------------------------
            # Inspection rejection
            # ------------------------------------------------

            if (
                final_result.status
                != "eligible"
            ):

                save_rejection(
                    conn,
                    row,
                    final_result,
                )

                print(
                    f"  -> REJECT "
                    f"{final_result.status} "
                    f"method="
                    f"{final_result.acquisition_method} "
                    f"time="
                    f"{final_result.elapsed_seconds:.1f}s "
                    f"size={size_text} "
                    f"free="
                    f"{free_gb:.1f}GB"
                )

                continue

            temp_path = Path(
                final_result.temp_path
            )

            if selected >= target:

                shutil.rmtree(
                    temp_path,
                    ignore_errors=True,
                )

                continue

            final_path = (
                repositories_root
                / cohort
                / str(
                    final_result.repo_id
                )
            )

            final_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if final_path.exists():

                shutil.rmtree(
                    final_path,
                    ignore_errors=True,
                )

            temp_path.replace(
                final_path
            )

            save_acceptance(
                conn=conn,
                row=row,
                result=final_result,
                final_path=final_path,
                selection_rank=rank,
            )

            selected += 1
            rank += 1

            print(
                f"  -> ACCEPT "
                f"({selected:,}/"
                f"{target:,}) "
                f"method="
                f"{final_result.acquisition_method} "
                f"python="
                f"{final_result.python_file_count:,} "
                f"time="
                f"{final_result.elapsed_seconds:.1f}s "
                f"size="
                f"{size_text} "
                f"free="
                f"{free_gb:.1f}GB"
            )

        # ----------------------------------------------------
        # Remove any uncommitted successful acquisitions
        # after target is reached.
        # ----------------------------------------------------

        if selected >= target:

            for acquisition in acquisitions.values():

                if (
                    acquisition.status
                    == "success"
                    and acquisition.temp_path
                    is not None
                ):

                    temp_path = Path(
                        acquisition.temp_path
                    )

                    if temp_path.exists():

                        shutil.rmtree(
                            temp_path,
                            ignore_errors=True,
                        )

            break

    print()

    print(
        f"{cohort}: "
        f"{selected:,}/"
        f"{target:,} selected"
    )

    if (
        pilot is None
        and selected < target
        and index >= len(
            pending_rows
        )
    ):

        print(
            "WARNING: all available "
            "preclone candidates were processed, "
            f"but only {selected:,}/"
            f"{target:,} repositories "
            "were obtained."
        )

    return True


# ============================================================
# Export
# ============================================================

def export_results(
    conn: sqlite3.Connection,
    candidates: pd.DataFrame,
    config: dict,
):

    output = config[
        "output"
    ]

    clone_results_path = Path(
        output[
            "clone_results"
        ]
    )

    selected_path = Path(
        output[
            "selected_repositories"
        ]
    )

    clone_results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df = pd.read_sql_query(
        """
        SELECT *
        FROM clone_results
        ORDER BY
            cohort,
            random_key
        """,
        conn,
    )

    result_df.to_csv(
        clone_results_path,
        index=False,
    )

    if result_df.empty:

        pd.DataFrame().to_csv(
            selected_path,
            index=False,
        )

        return

    selected_state = (
        result_df[
            result_df[
                "final_selected"
            ] == 1
        ][[
            "repo_id",
            "clone_status",
            "clone_path",
            "clone_elapsed_seconds",
            "clone_size_bytes",
            "python_file_count",
            "total_file_count",
            "acquisition_method",
            "archive_branch",
            "selection_rank",
        ]]
    )

    selected_df = (
        candidates.merge(
            selected_state,
            on="repo_id",
            how="inner",
        )
    )

    selected_df = (
        selected_df
        .sort_values([
            "cohort",
            "selection_rank",
        ])
        .reset_index(
            drop=True
        )
    )

    selected_df.to_csv(
        selected_path,
        index=False,
    )

    print()
    print(
        f"Saved: "
        f"{clone_results_path}"
    )

    print(
        f"Saved: "
        f"{selected_path}"
    )


# ============================================================
# Statistics
# ============================================================

def print_statistics(
    conn: sqlite3.Connection,
):

    df = pd.read_sql_query(
        """
        SELECT *
        FROM clone_results
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
        "CLONE / ARCHIVE STATISTICS"
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
        print(
            cohort
        )

        print(
            f"processed: "
            f"{len(g):,}"
        )

        print(
            "status:"
        )

        print(
            g[
                "clone_status"
            ].value_counts()
        )

        if (
            "acquisition_method"
            in g.columns
        ):

            print(
                "acquisition method:"
            )

            print(
                g[
                    "acquisition_method"
                ]
                .fillna("none")
                .value_counts()
            )

        times = (
            g[
                "clone_elapsed_seconds"
            ]
            .dropna()
        )

        if not times.empty:

            print(
                f"time median: "
                f"{times.median():.2f}s"
            )

            print(
                f"time mean:   "
                f"{times.mean():.2f}s"
            )

            print(
                f"time p95:    "
                f"{times.quantile(.95):.2f}s"
            )

        sizes = (
            g[
                "clone_size_bytes"
            ]
            .dropna()
        )

        if not sizes.empty:

            print(
                "size median: "
                f"{sizes.median() / 1024 / 1024:.2f} MB"
            )

            print(
                "size mean:   "
                f"{sizes.mean() / 1024 / 1024:.2f} MB"
            )

            print(
                "size p95:    "
                f"{sizes.quantile(.95) / 1024 / 1024:.2f} MB"
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
            "Process at most N new "
            "repository candidates per cohort."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Override configured number "
            "of concurrent workers."
        ),
    )

    args = parser.parse_args()

    config = load_config()

    cloning = config[
        "cloning"
    ]

    output = config[
        "output"
    ]

    preclone_path = Path(
        output[
            "preclone_candidates"
        ]
    )

    state_path = Path(
        output[
            "clone_state"
        ]
    )

    repositories_root = Path(
        output[
            "repositories_dir"
        ]
    )

    temp_root = (
        repositories_root
        / ".tmp"
    )

    repositories_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Input from 02
    # --------------------------------------------------------

    candidates = pd.read_csv(
        preclone_path
    )

    candidates = (
        candidates
        .sort_values([
            "cohort",
            "random_key",
        ])
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    target = int(
        cloning[
            "target_per_cohort"
        ]
    )

    workers = int(
        args.workers
        if args.workers is not None
        else cloning.get(
            "workers",
            8,
        )
    )

    git_timeout_seconds = int(
        cloning.get(
            "timeout_seconds",
            300,
        )
    )

    archive_timeout_seconds = int(
        cloning.get(
            "archive_timeout_seconds",
            300,
        )
    )

    min_free_gb = float(
        cloning.get(
            "min_free_gb",
            100,
        )
    )

    cooldown_seconds = [
        int(x)
        for x in cloning.get(
            "cooldown_seconds",
            [
                120,
                300,
                600,
            ],
        )
    ]

    exclude_dirs = set(
        cloning[
            "exclude_dirs"
        ]
    )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    conn = connect_database(
        state_path
    )

    clean_incomplete_temp_dirs(
        temp_root
    )

    reconcile_database(
        conn,
        repositories_root,
    )

    started = time.time()

    print()
    print(
        "=" * 70
    )

    print(
        "REPOSITORY ACQUISITION"
    )

    print(
        "=" * 70
    )

    print(
        f"workers:          "
        f"{workers}"
    )

    print(
        f"git timeout:      "
        f"{git_timeout_seconds}s"
    )

    print(
        f"archive timeout:  "
        f"{archive_timeout_seconds}s"
    )

    print(
        f"minimum free:     "
        f"{min_free_gb:.1f} GB"
    )

    print(
        "cooldowns:        "
        + ", ".join(
            f"{x}s"
            for x
            in cooldown_seconds
        )
    )

    print(
        f"current free:     "
        f"{free_space_gb(repositories_root):.1f} GB"
    )

    if args.pilot is not None:

        print(
            f"PILOT:             "
            f"{args.pilot} "
            "new candidates per cohort"
        )

    stopped_safely = False

    try:

        for cohort in COHORTS:

            completed = process_cohort(
                conn=conn,
                candidates=candidates,
                cohort=cohort,
                repositories_root=(
                    repositories_root
                ),
                temp_root=temp_root,
                exclude_dirs=(
                    exclude_dirs
                ),
                workers=workers,
                git_timeout_seconds=(
                    git_timeout_seconds
                ),
                archive_timeout_seconds=(
                    archive_timeout_seconds
                ),
                min_free_gb=(
                    min_free_gb
                ),
                cooldown_seconds=(
                    cooldown_seconds
                ),
                target=target,
                pilot=args.pilot,
            )

            export_results(
                conn,
                candidates,
                config,
            )

            if not completed:

                stopped_safely = True
                break

    except KeyboardInterrupt:

        stopped_safely = True

        print()
        print()
        print(
            "Interrupted by user."
        )

        print(
            "Completed repositories "
            "remain saved."
        )

        print(
            "Temporary acquisitions "
            "will be removed automatically "
            "on the next run."
        )

    finally:

        export_results(
            conn,
            candidates,
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

        if stopped_safely:

            print(
                "ACQUISITION SAFELY STOPPED"
            )

        else:

            print(
                "ACQUISITION FINISHED"
            )

        print(
            "=" * 70
        )

        print(
            f"Elapsed: "
            f"{elapsed / 3600:.2f} hours"
        )

        print(
            f"Free disk space: "
            f"{free_space_gb(repositories_root):.1f} GB"
        )

        for cohort in COHORTS:

            count = selected_count(
                conn,
                cohort,
            )

            print(
                f"{cohort}: "
                f"{count:,}/"
                f"{target:,} selected"
            )

        conn.close()


if __name__ == "__main__":
    main()
