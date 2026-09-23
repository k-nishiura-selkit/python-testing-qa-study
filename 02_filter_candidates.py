from __future__ import annotations

import argparse
import ast
import hashlib
import html
import os
import re
import sqlite3
import time
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
# Deterministic random ordering
# ============================================================

def deterministic_random_key(
    seed: int,
    cohort: str,
    repo_id: int,
) -> str:

    value = (
        f"{seed}:{cohort}:{repo_id}"
    ).encode("utf-8")

    return hashlib.sha256(
        value
    ).hexdigest()


# ============================================================
# Text normalization
# ============================================================

def normalize_text(
    value: str | None,
) -> str:

    if value is None:
        return ""

    value = html.unescape(
        str(value)
    )

    value = value.lower()

    # hyphen / underscore / whitespace are
    # treated as equivalent separators.
    value = re.sub(
        r"[-_\s]+",
        " ",
        value,
    )

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def normalize_keyword(
    keyword: str,
) -> str:

    return normalize_text(
        keyword
    )


def contains_keyword(
    text: str | None,
    keywords: list[str],
) -> str | None:

    normalized = normalize_text(
        text
    )

    if not normalized:
        return None

    padded = (
        " "
        + normalized
        + " "
    )

    for keyword in keywords:

        k = normalize_keyword(
            keyword
        )

        if not k:
            continue

        target = (
            " "
            + k
            + " "
        )

        if target in padded:
            return keyword

    return None


# ============================================================
# Utility
# ============================================================

def clean_value(
    value,
):

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def bool_from_csv(
    value,
) -> bool | None:

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(
        value,
        bool,
    ):
        return value

    text = str(
        value
    ).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
    }:
        return False

    return None


# ============================================================
# Topics parsing
# ============================================================

def parse_topics(
    value,
) -> list[str]:

    if value is None:
        return []

    if isinstance(
        value,
        list,
    ):
        return [
            str(x)
            for x in value
        ]

    text = str(
        value
    ).strip()

    if not text:
        return []

    try:

        parsed = ast.literal_eval(
            text
        )

        if isinstance(
            parsed,
            list,
        ):
            return [
                str(x)
                for x in parsed
            ]

    except Exception:
        pass

    return [
        part.strip()
        for part in text.split(",")
        if part.strip()
    ]


def serialize_topics(
    value,
) -> str:

    return "|".join(
        parse_topics(
            value
        )
    )


# ============================================================
# Local metadata filtering
# ============================================================

def local_exclusion_reason(
    row: pd.Series,
    config: dict,
) -> tuple[
    str | None,
    str | None,
]:

    filtering = config[
        "filtering"
    ]

    # --------------------------------------------------------
    # Existing metadata decision from 01
    # --------------------------------------------------------

    eligible = bool_from_csv(
        row.get(
            "eligible_metadata"
        )
    )

    if eligible is False:

        previous_reason = row.get(
            "exclusion_reason"
        )

        return (
            "metadata_excluded",
            (
                str(previous_reason)
                if clean_value(
                    previous_reason
                ) is not None
                else None
            ),
        )

    # --------------------------------------------------------
    # Repository name
    # --------------------------------------------------------

    full_name = str(
        row.get(
            "full_name",
            "",
        )
    )

    repo_name = (
        full_name.split(
            "/",
            1,
        )[-1]
    )

    matched = contains_keyword(
        repo_name,
        filtering[
            "name_keywords"
        ],
    )

    if matched is not None:

        return (
            "name_keyword",
            matched,
        )

    # --------------------------------------------------------
    # Description
    # --------------------------------------------------------

    description = clean_value(
        row.get(
            "description"
        )
    )

    matched = contains_keyword(
        description,
        filtering[
            "text_keywords"
        ],
    )

    if matched is not None:

        return (
            "description_keyword",
            matched,
        )

    # --------------------------------------------------------
    # Topics
    # --------------------------------------------------------

    topics = parse_topics(
        row.get(
            "topics"
        )
    )

    topics_text = " ".join(
        topics
    )

    matched = contains_keyword(
        topics_text,
        filtering[
            "text_keywords"
        ],
    )

    if matched is not None:

        return (
            "topics_keyword",
            matched,
        )

    return (
        None,
        None,
    )


# ============================================================
# GitHub REST API client
# ============================================================

class GitHubRestClient:

    BASE_URL = (
        "https://api.github.com"
    )

    def __init__(
        self,
        api_version: str,
        interval_seconds: float,
        max_retries: int,
    ):

        token = os.environ.get(
            "GITHUB_TOKEN"
        )

        if not token:
            raise RuntimeError(
                "Environment variable "
                "GITHUB_TOKEN is not set."
            )

        self.token = token
        self.api_version = (
            api_version
        )
        self.interval_seconds = (
            interval_seconds
        )
        self.max_retries = (
            max_retries
        )

        self.last_request_start_time = 0.0
        self.request_count = 0

        self._reset_session()

    def _reset_session(
        self,
    ):

        if hasattr(
            self,
            "session",
        ):

            try:
                self.session.close()
            except Exception:
                pass

        self.session = (
            requests.Session()
        )

        self.session.headers.update({
            "Accept":
                "application/vnd.github+json",

            "Authorization":
                f"Bearer {self.token}",

            "X-GitHub-Api-Version":
                self.api_version,

            "User-Agent":
                "python-testing-foundation-study",
        })

    def _wait(
        self,
    ):

        elapsed = (
            time.time()
            - self.last_request_start_time
        )

        wait = (
            self.interval_seconds
            - elapsed
        )

        if wait > 0:
            time.sleep(
                wait
            )

    def get(
        self,
        path: str,
        params: dict | None = None,
        accept: str | None = None,
    ) -> requests.Response:

        for attempt in range(
            self.max_retries
        ):

            self._wait()

            self.last_request_start_time = (
                time.time()
            )

            headers = {}

            if accept is not None:
                headers[
                    "Accept"
                ] = accept

            try:

                response = (
                    self.session.get(
                        (
                            self.BASE_URL
                            + path
                        ),
                        params=params,
                        headers=headers,
                        timeout=60,
                    )
                )

                self.request_count += 1

            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:

                wait = min(
                    5 * (
                        2 ** attempt
                    ),
                    120,
                )

                print(
                    "  Network error: "
                    f"{type(e).__name__}; "
                    f"retry in {wait}s"
                )

                self._reset_session()

                time.sleep(
                    wait
                )

                continue

            if response.status_code in {
                200,
                204,
                404,
            }:
                return response

            # ------------------------------------------------
            # Rate limit
            # ------------------------------------------------

            if response.status_code in {
                403,
                429,
            }:

                retry_after = (
                    response.headers.get(
                        "Retry-After"
                    )
                )

                if retry_after:

                    wait = int(
                        retry_after
                    )

                else:

                    reset = (
                        response.headers.get(
                            "X-RateLimit-Reset"
                        )
                    )

                    if reset:

                        wait = max(
                            int(reset)
                            - int(
                                time.time()
                            ),
                            1,
                        )

                    else:
                        wait = 60

                print(
                    "  Rate limited; "
                    f"sleeping {wait}s"
                )

                time.sleep(
                    wait
                )

                continue

            # ------------------------------------------------
            # Server error
            # ------------------------------------------------

            if (
                500
                <= response.status_code
                < 600
            ):

                wait = min(
                    5 * (
                        2 ** attempt
                    ),
                    120,
                )

                print(
                    "  GitHub server error "
                    f"{response.status_code}; "
                    f"retry in {wait}s"
                )

                time.sleep(
                    wait
                )

                continue

            response.raise_for_status()

        raise RuntimeError(
            "GitHub REST request "
            "failed after retries: "
            f"{path}"
        )


# ============================================================
# Bot classification
# ============================================================

def is_explicit_bot(
    contributor: dict,
) -> bool:

    contributor_type = (
        contributor.get(
            "type"
        )
    )

    if contributor_type == "Bot":
        return True

    login = (
        contributor.get(
            "login"
        )
        or ""
    ).lower()

    if login.endswith(
        "[bot]"
    ):
        return True

    return False


# ============================================================
# Contributors
# ============================================================

def get_human_contributor_count(
    client: GitHubRestClient,
    full_name: str,
    max_humans: int,
) -> tuple[
    int | None,
    str,
]:

    owner, repo = (
        full_name.split(
            "/",
            1,
        )
    )

    human_count = 0
    page = 1

    while True:

        response = client.get(
            (
                f"/repos/{owner}/"
                f"{repo}/contributors"
            ),
            params={
                "anon":
                    "true",

                "per_page":
                    100,

                "page":
                    page,
            },
        )

        if response.status_code == 404:

            return (
                None,
                "not_found",
            )

        if response.status_code == 204:

            return (
                0,
                "empty_repository",
            )

        contributors = (
            response.json()
        )

        if not contributors:
            break

        for contributor in contributors:

            # Anonymous contributor cannot
            # be established as a bot, so it
            # is conservatively counted as human.
            if is_explicit_bot(
                contributor
            ):
                continue

            human_count += 1

            if (
                human_count
                > max_humans
            ):
                return (
                    human_count,
                    "over_limit",
                )

        if len(
            contributors
        ) < 100:
            break

        page += 1

    return (
        human_count,
        "ok",
    )


# ============================================================
# README
# ============================================================

BADGE_PATTERNS = [

    re.compile(
        r"^\s*!\[.*?\]\(.*?\)\s*$"
    ),

    re.compile(
        r"^\s*\[!\[.*?\]\(.*?\)\]\(.*?\)\s*$"
    ),

    re.compile(
        r"^\s*<img\b.*?>\s*$",
        re.IGNORECASE,
    ),
]


def remove_html_comments(
    text: str,
) -> str:

    return re.sub(
        r"<!--.*?-->",
        "",
        text,
        flags=re.DOTALL,
    )


def first_meaningful_readme_line(
    text: str,
    scan_lines: int,
) -> str | None:

    cleaned = remove_html_comments(
        text
    )

    lines = (
        cleaned.splitlines()
    )

    for raw_line in lines[
        :scan_lines
    ]:

        line = (
            raw_line.strip()
        )

        if not line:
            continue

        if any(
            pattern.match(
                line
            )
            for pattern
            in BADGE_PATTERNS
        ):
            continue

        # Remove Markdown heading markers,
        # but retain heading text.
        line = re.sub(
            r"^\s*#{1,6}\s*",
            "",
            line,
        ).strip()

        if not line:
            continue

        if re.fullmatch(
            r"[-*_]{3,}",
            line,
        ):
            continue

        return line

    return None


def get_readme_first_line(
    client: GitHubRestClient,
    full_name: str,
    default_branch: str | None,
    scan_lines: int,
) -> tuple[
    str | None,
    str,
]:

    owner, repo = (
        full_name.split(
            "/",
            1,
        )
    )

    params = {}

    if (
        default_branch
        and str(
            default_branch
        ).strip()
    ):

        params[
            "ref"
        ] = default_branch

    response = client.get(
        (
            f"/repos/{owner}/"
            f"{repo}/readme"
        ),
        params=params,
        accept=(
            "application/vnd.github.raw+json"
        ),
    )

    if response.status_code == 404:

        return (
            None,
            "missing",
        )

    text = (
        response.text
        or ""
    )

    first_line = (
        first_meaningful_readme_line(
            text,
            scan_lines,
        )
    )

    if first_line is None:

        return (
            None,
            "empty",
        )

    return (
        first_line,
        "ok",
    )


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

    return conn


def create_tables(
    conn: sqlite3.Connection,
):

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (

            repo_id INTEGER PRIMARY KEY,

            cohort TEXT NOT NULL,
            random_key TEXT NOT NULL,

            full_name TEXT NOT NULL,
            owner_login TEXT,
            owner_type TEXT,

            clone_url TEXT,
            default_branch TEXT,

            sampled_date TEXT,
            created_at TEXT,
            updated_at TEXT,
            pushed_at TEXT,

            size_kb INTEGER,
            stars INTEGER,
            forks_count INTEGER,
            open_issues_count INTEGER,

            archived INTEGER,
            license TEXT,

            description TEXT,
            topics TEXT,

            local_status TEXT NOT NULL,
            local_reason TEXT,
            local_detail TEXT,

            api_status TEXT,
            human_contributors INTEGER,

            readme_status TEXT,
            readme_first_line TEXT,
            readme_keyword TEXT,

            final_status TEXT,

            processed_at TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_candidates_order
        ON candidates(
            cohort,
            local_status,
            random_key
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_candidates_final
        ON candidates(
            cohort,
            final_status
        )
        """
    )

    conn.commit()


# ============================================================
# Import 01 output
# ============================================================

def import_candidates(
    conn: sqlite3.Connection,
    config: dict,
):

    candidates_dir = Path(
        config[
            "output"
        ][
            "candidates_dir"
        ]
    )

    seed = config[
        "filtering"
    ][
        "random_seed"
    ]

    print()
    print("=" * 70)
    print(
        "IMPORTING 01 CANDIDATES"
    )
    print("=" * 70)

    for cohort in COHORTS:

        cohort_dir = (
            candidates_dir
            / cohort
        )

        files = sorted(
            cohort_dir.glob(
                "*.csv"
            )
        )

        files = [
            path
            for path in files
            if not path.name.endswith(
                "__summary.csv"
            )
            and not path.name.endswith(
                "__partitions.csv"
            )
        ]

        print()
        print(
            f"{cohort}: "
            f"{len(files)} daily files"
        )

        imported = 0

        for file_index, path in enumerate(
            files,
            start=1,
        ):

            df = pd.read_csv(
                path
            )

            rows_to_insert = []

            for _, row in df.iterrows():

                repo_id = int(
                    row[
                        "repo_id"
                    ]
                )

                reason, detail = (
                    local_exclusion_reason(
                        row,
                        config,
                    )
                )

                if reason is None:

                    local_status = (
                        "eligible"
                    )

                else:

                    local_status = (
                        "excluded"
                    )

                random_key = (
                    deterministic_random_key(
                        seed,
                        cohort,
                        repo_id,
                    )
                )

                archived = bool_from_csv(
                    row.get(
                        "archived"
                    )
                )

                if archived is None:
                    archived_value = None
                else:
                    archived_value = int(
                        archived
                    )

                rows_to_insert.append((

                    repo_id,
                    cohort,
                    random_key,

                    clean_value(
                        row.get(
                            "full_name"
                        )
                    ),

                    clean_value(
                        row.get(
                            "owner_login"
                        )
                    ),

                    clean_value(
                        row.get(
                            "owner_type"
                        )
                    ),

                    clean_value(
                        row.get(
                            "clone_url"
                        )
                    ),

                    clean_value(
                        row.get(
                            "default_branch"
                        )
                    ),

                    clean_value(
                        row.get(
                            "sampled_date"
                        )
                    ),

                    clean_value(
                        row.get(
                            "created_at"
                        )
                    ),

                    clean_value(
                        row.get(
                            "updated_at"
                        )
                    ),

                    clean_value(
                        row.get(
                            "pushed_at"
                        )
                    ),

                    clean_value(
                        row.get(
                            "size_kb"
                        )
                    ),

                    clean_value(
                        row.get(
                            "stars"
                        )
                    ),

                    clean_value(
                        row.get(
                            "forks_count"
                        )
                    ),

                    clean_value(
                        row.get(
                            "open_issues_count"
                        )
                    ),

                    archived_value,

                    clean_value(
                        row.get(
                            "license"
                        )
                    ),

                    clean_value(
                        row.get(
                            "description"
                        )
                    ),

                    serialize_topics(
                        row.get(
                            "topics"
                        )
                    ),

                    local_status,
                    reason,
                    detail,
                ))

            conn.executemany(
                """
                INSERT OR IGNORE INTO candidates (

                    repo_id,
                    cohort,
                    random_key,

                    full_name,
                    owner_login,
                    owner_type,

                    clone_url,
                    default_branch,

                    sampled_date,
                    created_at,
                    updated_at,
                    pushed_at,

                    size_kb,
                    stars,
                    forks_count,
                    open_issues_count,

                    archived,
                    license,

                    description,
                    topics,

                    local_status,
                    local_reason,
                    local_detail

                )
                VALUES (
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?
                )
                """,
                rows_to_insert,
            )

            conn.commit()

            imported += len(
                rows_to_insert
            )

            print(
                f"\r  [{file_index}/"
                f"{len(files)}] "
                f"{imported:,} rows",
                end="",
                flush=True,
            )

        print()


# ============================================================
# Counts
# ============================================================

def count_status(
    conn: sqlite3.Connection,
    cohort: str,
    status: str,
) -> int:

    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM candidates
        WHERE cohort = ?
          AND final_status = ?
        """,
        (
            cohort,
            status,
        ),
    ).fetchone()

    return int(
        row[0]
    )


# ============================================================
# Process cohort
# ============================================================

def process_cohort(
    conn: sqlite3.Connection,
    client: GitHubRestClient,
    config: dict,
    cohort: str,
    target: int,
    pilot: int | None,
):

    filtering = config[
        "filtering"
    ]

    minimum = filtering[
        "min_human_contributors"
    ]

    maximum = filtering[
        "max_human_contributors"
    ]

    scan_lines = filtering[
        "readme_scan_lines"
    ]

    accepted = count_status(
        conn,
        cohort,
        "accepted",
    )

    print()
    print("=" * 70)

    print(
        f"{cohort}: "
        f"{accepted:,}/"
        f"{target:,} already accepted"
    )

    print("=" * 70)

    if accepted >= target:
        return

    query = """
        SELECT
            repo_id,
            full_name,
            clone_url,
            default_branch
        FROM candidates
        WHERE cohort = ?
          AND local_status = 'eligible'
          AND final_status IS NULL
        ORDER BY random_key
    """

    cursor = conn.execute(
        query,
        (
            cohort,
        ),
    )

    processed_this_run = 0

    for (
        repo_id,
        full_name,
        clone_url,
        default_branch,
    ) in cursor:

        if accepted >= target:
            break

        if (
            pilot is not None
            and processed_this_run
            >= pilot
        ):
            break

        processed_this_run += 1

        print(
            f"[{cohort}] "
            f"checked="
            f"{processed_this_run:,} "
            f"accepted="
            f"{accepted:,}/"
            f"{target:,} "
            f"repo="
            f"{full_name}"
        )

        # ----------------------------------------------------
        # Contributors
        # ----------------------------------------------------

        (
            human_count,
            contributor_status,
        ) = get_human_contributor_count(
            client=client,
            full_name=full_name,
            max_humans=maximum,
        )

        if contributor_status in {
            "not_found",
            "empty_repository",
        }:

            conn.execute(
                """
                UPDATE candidates
                SET
                    api_status = ?,
                    human_contributors = ?,
                    final_status = ?,
                    processed_at =
                        CURRENT_TIMESTAMP
                WHERE repo_id = ?
                """,
                (
                    contributor_status,
                    human_count,
                    (
                        "contributors_"
                        + contributor_status
                    ),
                    repo_id,
                ),
            )

            conn.commit()
            continue

        if (
            contributor_status
            == "over_limit"
        ):

            conn.execute(
                """
                UPDATE candidates
                SET
                    api_status = ?,
                    human_contributors = ?,
                    final_status =
                        'contributors_gt_max',
                    processed_at =
                        CURRENT_TIMESTAMP
                WHERE repo_id = ?
                """,
                (
                    contributor_status,
                    human_count,
                    repo_id,
                ),
            )

            conn.commit()
            continue

        if (
            human_count is None
            or human_count < minimum
        ):

            conn.execute(
                """
                UPDATE candidates
                SET
                    api_status = ?,
                    human_contributors = ?,
                    final_status =
                        'contributors_lt_min',
                    processed_at =
                        CURRENT_TIMESTAMP
                WHERE repo_id = ?
                """,
                (
                    contributor_status,
                    human_count,
                    repo_id,
                ),
            )

            conn.commit()
            continue

        # ----------------------------------------------------
        # README
        # ----------------------------------------------------

        (
            first_line,
            readme_status,
        ) = get_readme_first_line(
            client=client,
            full_name=full_name,
            default_branch=default_branch,
            scan_lines=scan_lines,
        )

        if readme_status != "ok":

            conn.execute(
                """
                UPDATE candidates
                SET
                    api_status = 'ok',
                    human_contributors = ?,
                    readme_status = ?,
                    final_status = ?,
                    processed_at =
                        CURRENT_TIMESTAMP
                WHERE repo_id = ?
                """,
                (
                    human_count,
                    readme_status,
                    (
                        "readme_"
                        + readme_status
                    ),
                    repo_id,
                ),
            )

            conn.commit()
            continue

        matched = contains_keyword(
            first_line,
            filtering[
                "text_keywords"
            ],
        )

        if matched is not None:

            conn.execute(
                """
                UPDATE candidates
                SET
                    api_status = 'ok',
                    human_contributors = ?,
                    readme_status = 'ok',
                    readme_first_line = ?,
                    readme_keyword = ?,
                    final_status =
                        'readme_keyword',
                    processed_at =
                        CURRENT_TIMESTAMP
                WHERE repo_id = ?
                """,
                (
                    human_count,
                    first_line,
                    matched,
                    repo_id,
                ),
            )

            conn.commit()
            continue

        # ----------------------------------------------------
        # Accepted for clone
        # ----------------------------------------------------

        conn.execute(
            """
            UPDATE candidates
            SET
                api_status = 'ok',
                human_contributors = ?,
                readme_status = 'ok',
                readme_first_line = ?,
                final_status = 'accepted',
                processed_at =
                    CURRENT_TIMESTAMP
            WHERE repo_id = ?
            """,
            (
                human_count,
                first_line,
                repo_id,
            ),
        )

        conn.commit()

        accepted += 1

        print(
            f"  -> ACCEPT "
            f"({accepted:,}/"
            f"{target:,})"
        )


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

    preclone_path = Path(
        output[
            "preclone_candidates"
        ]
    )

    summary_path = Path(
        output[
            "filter_summary"
        ]
    )

    preclone_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Accepted clone candidates
    # --------------------------------------------------------

    accepted_df = pd.read_sql_query(
        """
        SELECT

            repo_id,
            cohort,
            random_key,

            full_name,
            owner_login,
            owner_type,

            clone_url,
            default_branch,

            sampled_date,
            created_at,
            updated_at,
            pushed_at,

            size_kb,
            stars,
            forks_count,
            open_issues_count,

            archived,
            license,

            description,
            topics,

            human_contributors,

            readme_first_line

        FROM candidates

        WHERE final_status =
            'accepted'

        ORDER BY
            cohort,
            random_key
        """,
        conn,
    )

    accepted_df.to_csv(
        preclone_path,
        index=False,
    )

    # --------------------------------------------------------
    # Selection summary
    # --------------------------------------------------------

    summary_df = pd.read_sql_query(
        """
        SELECT

            cohort,

            CASE

                WHEN final_status
                    IS NOT NULL
                THEN final_status

                WHEN local_status =
                    'excluded'
                THEN local_reason

                ELSE
                    'not_processed'

            END AS status,

            COUNT(*) AS count

        FROM candidates

        GROUP BY
            cohort,
            status

        ORDER BY
            cohort,
            status
        """,
        conn,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        f"Saved: "
        f"{preclone_path}"
    )

    print(
        f"Saved: "
        f"{summary_path}"
    )


# ============================================================
# Detailed selection log
# ============================================================

def export_decisions(
    conn: sqlite3.Connection,
    path: Path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_sql_query(
        """
        SELECT *
        FROM candidates
        ORDER BY
            cohort,
            random_key
        """,
        conn,
    )

    df.to_csv(
        path,
        index=False,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--pilot",
        type=int,
        default=None,
        help=(
            "Process at most N "
            "API-stage candidates "
            "per cohort."
        ),
    )

    parser.add_argument(
        "--reimport",
        action="store_true",
        help=(
            "Rebuild SQLite state "
            "from 01 candidate CSVs. "
            "Existing API decisions "
            "are lost."
        ),
    )

    parser.add_argument(
        "--export-decisions",
        action="store_true",
        help=(
            "Export all candidate "
            "selection decisions "
            "to CSV."
        ),
    )

    args = (
        parser.parse_args()
    )

    config = (
        load_config()
    )

    output = config[
        "output"
    ]

    state_path = Path(
        output[
            "filter_state"
        ]
    )

    # --------------------------------------------------------
    # Rebuild state if explicitly requested
    # --------------------------------------------------------

    if (
        args.reimport
        and state_path.exists()
    ):

        state_path.unlink()

        wal = Path(
            str(state_path)
            + "-wal"
        )

        shm = Path(
            str(state_path)
            + "-shm"
        )

        if wal.exists():
            wal.unlink()

        if shm.exists():
            shm.unlink()

    database_was_new = (
        not state_path.exists()
    )

    conn = connect_database(
        state_path
    )

    if database_was_new:

        import_candidates(
            conn,
            config,
        )

    else:

        print(
            "Using existing filter state: "
            f"{state_path}"
        )

    # --------------------------------------------------------
    # GitHub REST client
    # --------------------------------------------------------

    client = GitHubRestClient(
        api_version=config[
            "github"
        ][
            "api_version"
        ],

        interval_seconds=config[
            "github"
        ].get(
            "core_interval_seconds",
            0.8,
        ),

        max_retries=config[
            "github"
        ].get(
            "max_retries",
            10,
        ),
    )

    target = config[
        "filtering"
    ][
        "preclone_target_per_cohort"
    ]

    started = (
        time.time()
    )

    # --------------------------------------------------------
    # Process cohorts
    # --------------------------------------------------------

    for cohort in COHORTS:

        process_cohort(
            conn=conn,
            client=client,
            config=config,
            cohort=cohort,
            target=target,
            pilot=args.pilot,
        )

        export_results(
            conn,
            config,
        )

    # --------------------------------------------------------
    # Final export
    # --------------------------------------------------------

    export_results(
        conn,
        config,
    )

    if args.export_decisions:

        decisions_path = (
            state_path.parent
            / "filter_decisions.csv"
        )

        export_decisions(
            conn,
            decisions_path,
        )

        print(
            f"Saved: "
            f"{decisions_path}"
        )

    elapsed = (
        time.time()
        - started
    )

    print()
    print("=" * 70)
    print(
        "FILTERING FINISHED"
    )
    print("=" * 70)

    print(
        "REST API requests "
        "this run: "
        f"{client.request_count:,}"
    )

    print(
        "Elapsed: "
        f"{elapsed / 3600:.2f} hours"
    )

    for cohort in COHORTS:

        accepted = (
            count_status(
                conn,
                cohort,
                "accepted",
            )
        )

        print(
            f"{cohort}: "
            f"{accepted:,}/"
            f"{target:,} accepted"
        )

    conn.close()


if __name__ == "__main__":
    main()
