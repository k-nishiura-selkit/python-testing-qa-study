from __future__ import annotations

import argparse
import math
import os
import random
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml


CONFIG_PATH = Path("config.yaml")

PER_PAGE = 100
MAX_SEARCH_RESULTS = 1000


# ============================================================
# Configuration
# ============================================================

def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# GitHub Search API client
# ============================================================

class GitHubSearchClient:

    BASE_URL = "https://api.github.com"

    def __init__(
        self,
        api_version: str,
        interval_seconds: float,
        max_retries: int,
    ):
        token = os.environ.get("GITHUB_TOKEN")

        if not token:
            raise RuntimeError(
                "Environment variable GITHUB_TOKEN is not set."
            )

        self.token = token
        self.api_version = api_version
        self.interval_seconds = interval_seconds
        self.max_retries = max_retries

        # リクエスト「開始時刻」の間隔を管理する
        self.last_request_start_time = 0.0
        self.request_count = 0

        self._reset_session()

    def _reset_session(self):

        if hasattr(self, "session"):
            try:
                self.session.close()
            except Exception:
                pass

        self.session = requests.Session()

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

    def _wait(self):

        elapsed = (
            time.time()
            - self.last_request_start_time
        )

        wait = (
            self.interval_seconds
            - elapsed
        )

        if wait > 0:
            time.sleep(wait)

    def search(
        self,
        query: str,
        page: int = 1,
    ) -> dict:

        for attempt in range(
            self.max_retries
        ):

            self._wait()

            # 開始時刻をGETの直前に記録
            self.last_request_start_time = (
                time.time()
            )

            try:

                response = self.session.get(
                    f"{self.BASE_URL}/search/repositories",
                    params={
                        "q": query,
                        "per_page": PER_PAGE,
                        "page": page,
                    },
                    timeout=60,
                )

                self.request_count += 1

            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as e:

                wait = min(
                    5 * (2 ** attempt),
                    120,
                )

                print(
                    f"    Network error: "
                    f"{type(e).__name__}"
                )

                print(
                    f"    Retry "
                    f"{attempt + 1}/"
                    f"{self.max_retries} "
                    f"in {wait}s"
                )

                self._reset_session()

                time.sleep(wait)
                continue

            # ------------------------------------------------
            # Success
            # ------------------------------------------------

            if response.status_code == 200:

                data = response.json()

                if data.get(
                    "incomplete_results",
                    False,
                ):

                    wait = min(
                        10 * (attempt + 1),
                        120,
                    )

                    print(
                        "    incomplete_results=True; "
                        f"retry in {wait}s"
                    )

                    time.sleep(wait)
                    continue

                return data

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
                            - int(time.time()),
                            1,
                        )

                    else:
                        wait = 60

                print(
                    f"    Rate limited; "
                    f"sleeping {wait}s"
                )

                time.sleep(wait)
                continue

            # ------------------------------------------------
            # GitHub server error
            # ------------------------------------------------

            if (
                500
                <= response.status_code
                < 600
            ):

                wait = min(
                    5 * (2 ** attempt),
                    120,
                )

                print(
                    "    GitHub server error "
                    f"{response.status_code}; "
                    f"retry in {wait}s"
                )

                time.sleep(wait)
                continue

            response.raise_for_status()

        raise RuntimeError(
            "GitHub request failed after "
            f"{self.max_retries} retries:\n"
            f"{query}"
        )


# ============================================================
# Date sampling
# ============================================================

def month_starts(
    start_date: date,
    end_date: date,
) -> list[date]:

    current = date(
        start_date.year,
        start_date.month,
        1,
    )

    result = []

    while current <= end_date:

        result.append(current)

        if current.month == 12:

            current = date(
                current.year + 1,
                1,
                1,
            )

        else:

            current = date(
                current.year,
                current.month + 1,
                1,
            )

    return result


def dates_in_month(
    month_start: date,
    overall_start: date,
    overall_end: date,
) -> list[date]:

    if month_start.month == 12:

        next_month = date(
            month_start.year + 1,
            1,
            1,
        )

    else:

        next_month = date(
            month_start.year,
            month_start.month + 1,
            1,
        )

    month_end = (
        next_month
        - timedelta(days=1)
    )

    start = max(
        month_start,
        overall_start,
    )

    end = min(
        month_end,
        overall_end,
    )

    return [
        start + timedelta(days=i)
        for i in range(
            (end - start).days + 1
        )
    ]


def sample_days(
    config: dict,
) -> pd.DataFrame:

    seed = config[
        "sampling"
    ]["random_seed"]

    days_per_month = config[
        "sampling"
    ]["days_per_month"]

    rows = []

    for cohort in [
        "pre_ai",
        "contemporary",
    ]:

        period = config[
            "periods"
        ][cohort]

        start = date.fromisoformat(
            period["start"]
        )

        end = date.fromisoformat(
            period["end"]
        )

        for month_start in month_starts(
            start,
            end,
        ):

            available = dates_in_month(
                month_start,
                start,
                end,
            )

            if days_per_month > len(available):
                raise ValueError(
                    f"{cohort} "
                    f"{month_start:%Y-%m}: "
                    "days_per_month exceeds "
                    "available days."
                )

            month_seed = (
                f"{seed}:"
                f"{cohort}:"
                f"{month_start:%Y-%m}"
            )

            rng = random.Random(
                month_seed
            )

            selected = rng.sample(
                available,
                days_per_month,
            )

            selected.sort()

            for d in selected:

                rows.append({
                    "cohort":
                        cohort,

                    "year":
                        d.year,

                    "month":
                        d.month,

                    "date":
                        d.isoformat(),
                })

    return pd.DataFrame(rows)


# ============================================================
# Search qualifier helpers
# ============================================================

def range_qualifier(
    name: str,
    minimum: int,
    maximum: int | None,
) -> str:

    if maximum is None:
        return f"{name}:>={minimum}"

    if minimum == maximum:
        return f"{name}:{minimum}"

    return (
        f"{name}:{minimum}..{maximum}"
    )


def github_datetime(
    dt: datetime,
) -> str:

    return (
        dt.astimezone(
            timezone.utc
        )
        .strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )


def build_query(
    language: str,
    target_date: date,
    exclude_forks: bool,
    size_min: int,
    size_max: int | None,
    stars_min: int | None = None,
    stars_max: int | None = None,
    created_start: datetime | None = None,
    created_end: datetime | None = None,
) -> str:

    parts = [
        f"language:{language}",
    ]

    # 通常は日単位。
    # 最終fallbackでは同日の時刻範囲を指定する。
    if (
        created_start is not None
        and created_end is not None
    ):

        parts.append(
            "created:"
            f"{github_datetime(created_start)}"
            ".."
            f"{github_datetime(created_end)}"
        )

    else:

        parts.append(
            f"created:{target_date.isoformat()}"
        )

    parts.append(
        range_qualifier(
            "size",
            size_min,
            size_max,
        )
    )

    if exclude_forks:
        parts.append(
            "fork:false"
        )

    if stars_min is not None:

        parts.append(
            range_qualifier(
                "stars",
                stars_min,
                stars_max,
            )
        )

    return " ".join(parts)


def build_base_query(
    language: str,
    target_date: date,
    exclude_forks: bool,
) -> str:

    parts = [
        f"language:{language}",
        f"created:{target_date.isoformat()}",
    ]

    if exclude_forks:
        parts.append(
            "fork:false"
        )

    return " ".join(parts)


# ============================================================
# Repository normalization
# ============================================================

def normalize(
    repo: dict,
    cohort: str,
    sampled_date: date,
) -> dict:

    owner = (
        repo.get("owner")
        or {}
    )

    license_info = (
        repo.get("license")
        or {}
    )

    return {
        "repo_id":
            repo["id"],

        "node_id":
            repo.get("node_id"),

        "full_name":
            repo.get("full_name"),

        "owner_login":
            owner.get("login"),

        "owner_type":
            owner.get("type"),

        "html_url":
            repo.get("html_url"),

        "clone_url":
            repo.get("clone_url"),

        "default_branch":
            repo.get(
                "default_branch"
            ),

        "created_at":
            repo.get("created_at"),

        "updated_at":
            repo.get("updated_at"),

        "pushed_at":
            repo.get("pushed_at"),

        "fork":
            repo.get("fork"),

        "archived":
            repo.get("archived"),

        "disabled":
            repo.get("disabled"),

        "is_template":
            repo.get("is_template"),

        "size_kb":
            repo.get("size"),

        "stars":
            repo.get(
                "stargazers_count"
            ),

        "forks_count":
            repo.get(
                "forks_count"
            ),

        "open_issues_count":
            repo.get(
                "open_issues_count"
            ),

        "language":
            repo.get("language"),

        "description":
            repo.get("description"),

        "topics":
            repo.get("topics"),

        "license":
            license_info.get(
                "spdx_id"
            ),

        "visibility":
            repo.get("visibility"),

        "cohort":
            cohort,

        "sampled_date":
            sampled_date.isoformat(),
    }


# ============================================================
# Metadata eligibility
# ============================================================

def metadata_exclusion_reason(
    row: pd.Series,
    config: dict,
) -> str | None:

    selection = config[
        "selection"
    ]

    if row.get("fork") is True:
        return "fork"

    if (
        selection.get(
            "exclude_templates",
            True,
        )
        and row.get(
            "is_template"
        ) is True
    ):
        return "template"

    if (
        selection.get(
            "exclude_disabled",
            True,
        )
        and row.get(
            "disabled"
        ) is True
    ):
        return "disabled"

    return None


def add_metadata_filter_columns(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:

    df = df.copy()

    if df.empty:

        df[
            "exclusion_reason"
        ] = pd.Series(
            dtype="object"
        )

        df[
            "eligible_metadata"
        ] = pd.Series(
            dtype="bool"
        )

        return df

    df[
        "exclusion_reason"
    ] = df.apply(
        lambda row:
            metadata_exclusion_reason(
                row,
                config,
            ),
        axis=1,
    )

    df[
        "eligible_metadata"
    ] = df[
        "exclusion_reason"
    ].isna()

    return df


# ============================================================
# Partition helpers
# ============================================================

def range_label(
    minimum: int,
    maximum: int | None,
) -> str:

    if maximum is None:
        return f"{minimum}+"

    if minimum == maximum:
        return str(minimum)

    return (
        f"{minimum}-{maximum}"
    )


def split_range(
    minimum: int,
    maximum: int,
) -> tuple[
    tuple[int, int],
    tuple[int, int],
]:

    if minimum >= maximum:
        raise ValueError(
            "Cannot split range."
        )

    midpoint = (
        minimum + maximum
    ) // 2

    return (
        (
            minimum,
            midpoint,
        ),
        (
            midpoint + 1,
            maximum,
        ),
    )


# ============================================================
# Fetch safe leaf (<=1000)
# ============================================================

def fetch_leaf(
    client: GitHubSearchClient,
    query: str,
    first_response: dict,
    cohort: str,
    target_date: date,
) -> list[dict]:

    total_count = int(
        first_response.get(
            "total_count",
            0,
        )
    )

    if total_count > MAX_SEARCH_RESULTS:

        raise RuntimeError(
            "Unsafe leaf with "
            f"{total_count} results."
        )

    rows = [
        normalize(
            repo,
            cohort,
            target_date,
        )
        for repo in first_response.get(
            "items",
            [],
        )
    ]

    pages = math.ceil(
        total_count
        / PER_PAGE
    )

    for page in range(
        2,
        pages + 1,
    ):

        response = client.search(
            query,
            page=page,
        )

        rows.extend(
            normalize(
                repo,
                cohort,
                target_date,
            )
            for repo in response.get(
                "items",
                [],
            )
        )

    return rows


# ============================================================
# created_at time fallback
# ============================================================

def collect_created_partition(
    client: GitHubSearchClient,
    config: dict,
    cohort: str,
    target_date: date,
    size_value: int,
    stars_value: int,
    created_start: datetime,
    created_end: datetime,
    depth: int,
    partition_records: list[dict],
) -> list[dict]:

    selection = config[
        "selection"
    ]

    query = build_query(
        language=selection[
            "language"
        ],
        target_date=target_date,
        exclude_forks=selection.get(
            "exclude_forks",
            True,
        ),
        size_min=size_value,
        size_max=size_value,
        stars_min=stars_value,
        stars_max=stars_value,
        created_start=created_start,
        created_end=created_end,
    )

    response = client.search(
        query,
        page=1,
    )

    total_count = int(
        response.get(
            "total_count",
            0,
        )
    )

    indent = "  " * depth

    print(
        f"{indent}"
        f"size={size_value:<7} "
        f"stars={stars_value:<7} "
        f"created="
        f"{github_datetime(created_start)}"
        f".."
        f"{github_datetime(created_end)} "
        f"count={total_count:,}"
    )

    # --------------------------------------------------------
    # Safe leaf
    # --------------------------------------------------------

    if total_count <= MAX_SEARCH_RESULTS:

        rows = fetch_leaf(
            client=client,
            query=query,
            first_response=response,
            cohort=cohort,
            target_date=target_date,
        )

        partition_records.append({
            "cohort":
                cohort,

            "date":
                target_date.isoformat(),

            "size_min":
                size_value,

            "size_max":
                size_value,

            "stars_min":
                stars_value,

            "stars_max":
                stars_value,

            "created_start":
                github_datetime(
                    created_start
                ),

            "created_end":
                github_datetime(
                    created_end
                ),

            "reported_count":
                total_count,

            "retrieved_count":
                len(rows),

            "depth":
                depth,

            "partition_axis":
                "created",

            "status":
                "leaf",
        })

        return rows

    # --------------------------------------------------------
    # Can no longer split below one second
    # --------------------------------------------------------

    interval_seconds = int(
        (
            created_end
            - created_start
        ).total_seconds()
    )

    if interval_seconds <= 0:

        raise RuntimeError(
            "More than 1000 repositories "
            "have the exact same size, "
            "star count, and creation second:\n"
            f"{query}\n"
            f"count={total_count}"
        )

    # --------------------------------------------------------
    # Split creation-time interval
    # --------------------------------------------------------

    half = (
        interval_seconds // 2
    )

    midpoint = (
        created_start
        + timedelta(
            seconds=half
        )
    )

    right_start = (
        midpoint
        + timedelta(seconds=1)
    )

    print(
        f"{indent}  -> split created time"
    )

    rows = []

    rows.extend(
        collect_created_partition(
            client=client,
            config=config,
            cohort=cohort,
            target_date=target_date,
            size_value=size_value,
            stars_value=stars_value,
            created_start=created_start,
            created_end=midpoint,
            depth=depth + 1,
            partition_records=partition_records,
        )
    )

    if right_start <= created_end:

        rows.extend(
            collect_created_partition(
                client=client,
                config=config,
                cohort=cohort,
                target_date=target_date,
                size_value=size_value,
                stars_value=stars_value,
                created_start=right_start,
                created_end=created_end,
                depth=depth + 1,
                partition_records=partition_records,
            )
        )

    return rows


# ============================================================
# Stars fallback
# ============================================================

def collect_star_partition(
    client: GitHubSearchClient,
    config: dict,
    cohort: str,
    target_date: date,
    size_value: int,
    stars_min: int,
    stars_max: int | None,
    depth: int,
    partition_records: list[dict],
) -> list[dict]:

    selection = config[
        "selection"
    ]

    query = build_query(
        language=selection[
            "language"
        ],
        target_date=target_date,
        exclude_forks=selection.get(
            "exclude_forks",
            True,
        ),
        size_min=size_value,
        size_max=size_value,
        stars_min=stars_min,
        stars_max=stars_max,
    )

    response = client.search(
        query,
        page=1,
    )

    total_count = int(
        response.get(
            "total_count",
            0,
        )
    )

    indent = "  " * depth

    print(
        f"{indent}"
        f"size={size_value:<8} "
        f"stars="
        f"{range_label(stars_min, stars_max):<12} "
        f"count={total_count:,}"
    )

    # --------------------------------------------------------
    # Safe leaf
    # --------------------------------------------------------

    if total_count <= MAX_SEARCH_RESULTS:

        rows = fetch_leaf(
            client=client,
            query=query,
            first_response=response,
            cohort=cohort,
            target_date=target_date,
        )

        partition_records.append({
            "cohort":
                cohort,

            "date":
                target_date.isoformat(),

            "size_min":
                size_value,

            "size_max":
                size_value,

            "stars_min":
                stars_min,

            "stars_max":
                stars_max,

            "created_start":
                None,

            "created_end":
                None,

            "reported_count":
                total_count,

            "retrieved_count":
                len(rows),

            "depth":
                depth,

            "partition_axis":
                "stars",

            "status":
                "leaf",
        })

        return rows

    # --------------------------------------------------------
    # Finite star range can still be split
    # --------------------------------------------------------

    if (
        stars_max is not None
        and stars_min < stars_max
    ):

        print(
            f"{indent}  -> split stars"
        )

        rows = []

        for (
            child_min,
            child_max,
        ) in split_range(
            stars_min,
            stars_max,
        ):

            rows.extend(
                collect_star_partition(
                    client=client,
                    config=config,
                    cohort=cohort,
                    target_date=target_date,
                    size_value=size_value,
                    stars_min=child_min,
                    stars_max=child_max,
                    depth=depth + 1,
                    partition_records=partition_records,
                )
            )

        return rows

    # --------------------------------------------------------
    # Exact stars still >1000
    # -> creation-time fallback
    # --------------------------------------------------------

    if (
        stars_max is not None
        and stars_min == stars_max
    ):

        print(
            f"{indent}  -> exact size and stars "
            "still >1000; fallback to created time"
        )

        created_start = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            0,
            0,
            0,
            tzinfo=timezone.utc,
        )

        created_end = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            23,
            59,
            59,
            tzinfo=timezone.utc,
        )

        return collect_created_partition(
            client=client,
            config=config,
            cohort=cohort,
            target_date=target_date,
            size_value=size_value,
            stars_value=stars_min,
            created_start=created_start,
            created_end=created_end,
            depth=depth + 1,
            partition_records=partition_records,
        )

    # --------------------------------------------------------
    # Open-ended stars bucket still >1000
    # --------------------------------------------------------

    if stars_max is None:

        # 例: stars >= 1001
        # 1001-2001 / 2002+ のように分割する。
        finite_end = (
            stars_min * 2 - 1
        )

        print(
            f"{indent}  -> split open-ended stars"
        )

        rows = []

        rows.extend(
            collect_star_partition(
                client=client,
                config=config,
                cohort=cohort,
                target_date=target_date,
                size_value=size_value,
                stars_min=stars_min,
                stars_max=finite_end,
                depth=depth + 1,
                partition_records=partition_records,
            )
        )

        rows.extend(
            collect_star_partition(
                client=client,
                config=config,
                cohort=cohort,
                target_date=target_date,
                size_value=size_value,
                stars_min=finite_end + 1,
                stars_max=None,
                depth=depth + 1,
                partition_records=partition_records,
            )
        )

        return rows

    raise RuntimeError(
        "Unexpected star partition state."
    )


# ============================================================
# Size-first recursive partition
# ============================================================

def collect_size_partition(
    client: GitHubSearchClient,
    config: dict,
    cohort: str,
    target_date: date,
    size_min: int,
    size_max: int | None,
    depth: int,
    partition_records: list[dict],
) -> list[dict]:

    selection = config[
        "selection"
    ]

    query = build_query(
        language=selection[
            "language"
        ],
        target_date=target_date,
        exclude_forks=selection.get(
            "exclude_forks",
            True,
        ),
        size_min=size_min,
        size_max=size_max,
    )

    response = client.search(
        query,
        page=1,
    )

    total_count = int(
        response.get(
            "total_count",
            0,
        )
    )

    indent = "  " * depth

    print(
        f"{indent}"
        f"size="
        f"{range_label(size_min, size_max):<15} "
        f"count={total_count:,}"
    )

    # --------------------------------------------------------
    # Safe leaf
    # --------------------------------------------------------

    if total_count <= MAX_SEARCH_RESULTS:

        rows = fetch_leaf(
            client=client,
            query=query,
            first_response=response,
            cohort=cohort,
            target_date=target_date,
        )

        partition_records.append({
            "cohort":
                cohort,

            "date":
                target_date.isoformat(),

            "size_min":
                size_min,

            "size_max":
                size_max,

            "stars_min":
                None,

            "stars_max":
                None,

            "created_start":
                None,

            "created_end":
                None,

            "reported_count":
                total_count,

            "retrieved_count":
                len(rows),

            "depth":
                depth,

            "partition_axis":
                "size",

            "status":
                "leaf",
        })

        return rows

    # --------------------------------------------------------
    # Finite size range: split size first
    # --------------------------------------------------------

    if (
        size_max is not None
        and size_min < size_max
    ):

        print(
            f"{indent}  -> split size"
        )

        rows = []

        for (
            child_min,
            child_max,
        ) in split_range(
            size_min,
            size_max,
        ):

            rows.extend(
                collect_size_partition(
                    client=client,
                    config=config,
                    cohort=cohort,
                    target_date=target_date,
                    size_min=child_min,
                    size_max=child_max,
                    depth=depth + 1,
                    partition_records=partition_records,
                )
            )

        return rows

    # --------------------------------------------------------
    # Exact size still >1000
    # -> stars fallback
    # --------------------------------------------------------

    if (
        size_max is not None
        and size_min == size_max
    ):

        print(
            f"{indent}  -> exact size still >1000; "
            "fallback to stars"
        )

        rows = []

        for (
            stars_min,
            stars_max,
        ) in config[
            "partition"
        ]["star_buckets"]:

            rows.extend(
                collect_star_partition(
                    client=client,
                    config=config,
                    cohort=cohort,
                    target_date=target_date,
                    size_value=size_min,
                    stars_min=stars_min,
                    stars_max=stars_max,
                    depth=depth + 1,
                    partition_records=partition_records,
                )
            )

        return rows

    # --------------------------------------------------------
    # Open-ended size bucket
    # --------------------------------------------------------

    if size_max is None:

        finite_end = (
            size_min * 2 - 1
        )

        print(
            f"{indent}  -> split open-ended size"
        )

        rows = []

        rows.extend(
            collect_size_partition(
                client=client,
                config=config,
                cohort=cohort,
                target_date=target_date,
                size_min=size_min,
                size_max=finite_end,
                depth=depth + 1,
                partition_records=partition_records,
            )
        )

        rows.extend(
            collect_size_partition(
                client=client,
                config=config,
                cohort=cohort,
                target_date=target_date,
                size_min=finite_end + 1,
                size_max=None,
                depth=depth + 1,
                partition_records=partition_records,
            )
        )

        return rows

    raise RuntimeError(
        "Unexpected size partition state."
    )


# ============================================================
# One complete day
# ============================================================

def collect_day(
    client: GitHubSearchClient,
    config: dict,
    cohort: str,
    target_date: date,
) -> tuple[
    pd.DataFrame,
    dict,
    list[dict],
]:

    selection = config[
        "selection"
    ]

    base_query = build_base_query(
        language=selection[
            "language"
        ],
        target_date=target_date,
        exclude_forks=selection.get(
            "exclude_forks",
            True,
        ),
    )

    request_start = (
        client.request_count
    )

    initial_response = client.search(
        base_query,
        page=1,
    )

    reported_count = int(
        initial_response.get(
            "total_count",
            0,
        )
    )

    print(
        "GitHub reported count: "
        f"{reported_count:,}"
    )

    partition_records = []
    rows = []

    for (
        size_min,
        size_max,
    ) in config[
        "partition"
    ][
        "initial_size_buckets"
    ]:

        print()

        print(
            "Initial size bucket: "
            f"{range_label(size_min, size_max)}"
        )

        rows.extend(
            collect_size_partition(
                client=client,
                config=config,
                cohort=cohort,
                target_date=target_date,
                size_min=size_min,
                size_max=size_max,
                depth=0,
                partition_records=partition_records,
            )
        )

    df = pd.DataFrame(
        rows
    )

    raw_count = len(
        df
    )

    if not df.empty:

        df = (
            df.drop_duplicates(
                subset=[
                    "repo_id"
                ]
            )
            .reset_index(
                drop=True
            )
        )

    collected_count = len(
        df
    )

    duplicate_count = (
        raw_count
        - collected_count
    )

    if reported_count > 0:

        coverage = (
            collected_count
            / reported_count
        )

    else:
        coverage = None

    requests_for_day = (
        client.request_count
        - request_start
    )

    summary = {
        "cohort":
            cohort,

        "date":
            target_date.isoformat(),

        "reported_count":
            reported_count,

        "raw_retrieved_count":
            raw_count,

        "collected_count":
            collected_count,

        "duplicate_count":
            duplicate_count,

        "coverage":
            coverage,

        "method":
            "size_stars_created_partition",

        "search_api_requests":
            requests_for_day,

        "status":
            "completed",
    }

    return (
        df,
        summary,
        partition_records,
    )


# ============================================================
# Global summaries
# ============================================================

def rebuild_global_summaries(
    candidates_dir: Path,
    summary_path: Path,
    partition_summary_path: Path,
):

    daily_summaries = []
    partition_summaries = []

    for cohort in [
        "pre_ai",
        "contemporary",
    ]:

        cohort_dir = (
            candidates_dir
            / cohort
        )

        if not cohort_dir.exists():
            continue

        for path in sorted(
            cohort_dir.glob(
                "*__summary.csv"
            )
        ):

            daily_summaries.append(
                pd.read_csv(path)
            )

        for path in sorted(
            cohort_dir.glob(
                "*__partitions.csv"
            )
        ):

            partition_summaries.append(
                pd.read_csv(path)
            )

    if daily_summaries:

        summary_df = pd.concat(
            daily_summaries,
            ignore_index=True,
        )

        summary_df = (
            summary_df
            .sort_values([
                "cohort",
                "date",
            ])
            .reset_index(
                drop=True
            )
        )

        summary_df.to_csv(
            summary_path,
            index=False,
        )

    if partition_summaries:

        partition_df = pd.concat(
            partition_summaries,
            ignore_index=True,
        )

        sort_columns = [
            "cohort",
            "date",
            "size_min",
            "stars_min",
            "created_start",
        ]

        sort_columns = [
            col
            for col in sort_columns
            if col in partition_df.columns
        ]

        partition_df = (
            partition_df
            .sort_values(
                sort_columns,
                na_position="first",
            )
            .reset_index(
                drop=True
            )
        )

        partition_df.to_csv(
            partition_summary_path,
            index=False,
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit-days",
        type=int,
        default=None,
        help=(
            "Testing only: process at most "
            "this many days per cohort."
        ),
    )

    args = parser.parse_args()

    config = load_config()

    output = config[
        "output"
    ]

    sampled_days_path = Path(
        output[
            "sampled_days"
        ]
    )

    summary_path = Path(
        output[
            "collection_summary"
        ]
    )

    partition_summary_path = Path(
        output[
            "partition_summary"
        ]
    )

    candidates_dir = Path(
        output[
            "candidates_dir"
        ]
    )

    sampled_days_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Generate / reuse sampled dates
    # ========================================================

    if sampled_days_path.exists():

        sampled_days = (
            pd.read_csv(
                sampled_days_path
            )
        )

        print(
            "Using existing "
            "sampled_days.csv"
        )

    else:

        sampled_days = sample_days(
            config
        )

        sampled_days.to_csv(
            sampled_days_path,
            index=False,
        )

        print(
            "Generated "
            "sampled_days.csv"
        )

    processing_days = (
        sampled_days.copy()
    )

    if args.limit_days is not None:

        processing_days = (
            processing_days
            .groupby(
                "cohort",
                group_keys=False,
            )
            .head(
                args.limit_days
            )
            .reset_index(
                drop=True
            )
        )

        print(
            f"TEST MODE: "
            f"{args.limit_days} "
            "day(s) per cohort"
        )

    # ========================================================
    # API client
    # ========================================================

    client = GitHubSearchClient(
        api_version=config[
            "github"
        ][
            "api_version"
        ],

        interval_seconds=config[
            "github"
        ][
            "search_interval_seconds"
        ],

        max_retries=config[
            "github"
        ].get(
            "max_retries",
            10,
        ),
    )

    total_days = len(
        processing_days
    )

    started = (
        time.time()
    )

    # ========================================================
    # Process sampled days
    # ========================================================

    for position, row in enumerate(
        processing_days.itertuples(
            index=False
        ),
        start=1,
    ):

        cohort = (
            row.cohort
        )

        target_date = (
            date.fromisoformat(
                row.date
            )
        )

        cohort_dir = (
            candidates_dir
            / cohort
        )

        cohort_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        candidate_file = (
            cohort_dir
            / f"{target_date}.csv"
        )

        daily_summary_file = (
            cohort_dir
            / (
                f"{target_date}"
                "__summary.csv"
            )
        )

        daily_partition_file = (
            cohort_dir
            / (
                f"{target_date}"
                "__partitions.csv"
            )
        )

        print()
        print(
            "=" * 70
        )

        print(
            f"[{position}/"
            f"{total_days}] "
            f"{cohort}: "
            f"{target_date}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Completed checkpoint
        # ----------------------------------------------------

        if (
            candidate_file.exists()
            and daily_summary_file.exists()
            and daily_partition_file.exists()
        ):

            print(
                "Using existing completed result."
            )

            continue

        day_started = (
            time.time()
        )

        try:

            (
                day_df,
                day_summary,
                partition_records,
            ) = collect_day(
                client=client,
                config=config,
                cohort=cohort,
                target_date=target_date,
            )

            # ------------------------------------------------
            # Metadata filtering
            # ------------------------------------------------

            day_df = (
                add_metadata_filter_columns(
                    day_df,
                    config,
                )
            )

            eligible_count = int(
                day_df[
                    "eligible_metadata"
                ].sum()
            )

            day_summary[
                "metadata_eligible_count"
            ] = eligible_count

            day_summary[
                "elapsed_seconds"
            ] = (
                time.time()
                - day_started
            )

            # ------------------------------------------------
            # Save only after entire day is complete
            # ------------------------------------------------

            day_df.to_csv(
                candidate_file,
                index=False,
            )

            pd.DataFrame(
                partition_records
            ).to_csv(
                daily_partition_file,
                index=False,
            )

            pd.DataFrame(
                [day_summary]
            ).to_csv(
                daily_summary_file,
                index=False,
            )

            print()

            print(
                "DAY TOTAL: "
                f"{len(day_df):,}"
            )

            print(
                "METADATA ELIGIBLE: "
                f"{eligible_count:,}"
            )

            if (
                day_summary[
                    "coverage"
                ]
                is not None
            ):

                print(
                    "COVERAGE: "
                    f"{day_summary['coverage']:.6f}"
                )

            print(
                "REQUESTS: "
                f"{day_summary['search_api_requests']:,}"
            )

            print(
                "ELAPSED: "
                f"{day_summary['elapsed_seconds'] / 60:.1f} min"
            )

        except Exception as e:

            print()
            print(
                "!" * 70
            )

            print(
                f"FAILED: "
                f"{cohort} "
                f"{target_date}"
            )

            print(
                f"{type(e).__name__}: "
                f"{e}"
            )

            print(
                "No completed checkpoint "
                "was written for this date."
            )

            print(
                "!" * 70
            )

            client._reset_session()

            continue

        # ----------------------------------------------------
        # Rebuild summaries after every completed day
        # ----------------------------------------------------

        rebuild_global_summaries(
            candidates_dir,
            summary_path,
            partition_summary_path,
        )

    # ========================================================
    # Final rebuild
    # ========================================================

    rebuild_global_summaries(
        candidates_dir,
        summary_path,
        partition_summary_path,
    )

    elapsed = (
        time.time()
        - started
    )

    print()
    print(
        "=" * 70
    )

    print(
        "COLLECTION FINISHED"
    )

    print(
        "=" * 70
    )

    print(
        "Search API requests "
        "this run: "
        f"{client.request_count:,}"
    )

    print(
        "Elapsed: "
        f"{elapsed / 3600:.2f} hours"
    )


if __name__ == "__main__":
    main()
