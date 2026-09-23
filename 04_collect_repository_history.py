#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests

INPUT = Path("data/processed/selected_repositories.csv")
OUTPUT = Path("data/processed/03b_repository_history.csv")

API = "https://api.github.com"
API_VERSION = "2022-11-28"
COHORTS = ["pre_ai", "contemporary"]

TERMINAL = {
    "ok",
    "single_commit",
    "empty_repository",
    "not_found",
    "branch_not_found",
}

FIELDS = [
    "repo_id",
    "cohort",
    "full_name",
    "default_branch",
    "history_status",
    "commit_count",
    "first_commit_date",
    "last_commit_date",
    "development_span_days",
    "is_single_commit",
    "api_requests",
    "http_status",
    "error_message",
    "collected_at",
]

RATE_RESERVE = 50
MAX_ATTEMPTS = 7
TIMEOUT = 30
REQUEST_DELAY = 0.05


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def duration(seconds):
    seconds = max(0, int(round(seconds)))
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def token():
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value.strip()

    try:
        p = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip()
    except Exception:
        pass

    return None


def commit_date(obj):
    commit = obj.get("commit") or {}
    committer = commit.get("committer") or {}
    author = commit.get("author") or {}
    return committer.get("date") or author.get("date")


def parse_date(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def last_page(link):
    if not link:
        return None

    for part in link.split(","):
        if 'rel="last"' not in part:
            continue
        m = re.search(r"[?&]page=(\d+)", part)
        if m:
            return int(m.group(1))

    return None


def retry_after(response):
    value = response.headers.get("Retry-After")
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                dt = parsedate_to_datetime(value)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return max(
                    0.0,
                    (dt - datetime.now(timezone.utc)).total_seconds(),
                )
            except Exception:
                pass

    if response.headers.get("X-RateLimit-Remaining") == "0":
        try:
            reset = int(response.headers["X-RateLimit-Reset"])
            return max(0.0, reset - time.time() + 5)
        except Exception:
            pass

    return None


class GitHub:
    def __init__(self, auth_token, delay):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {auth_token}",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "python-testing-study-history",
        })
        self.delay = max(0.0, delay)
        self.requests = 0

    def maybe_wait_primary(self, response):
        try:
            remaining = int(
                response.headers["X-RateLimit-Remaining"]
            )
            reset = int(
                response.headers["X-RateLimit-Reset"]
            )
        except Exception:
            return

        if remaining > RATE_RESERVE:
            return

        wait = max(0.0, reset - time.time() + 5)

        print(
            "\n" + "!" * 68
            + "\nPRIMARY RATE LIMIT NEAR EXHAUSTION"
            + f"\nremaining: {remaining}"
            + f"\nwaiting: {duration(wait)}"
            + "\n" + "!" * 68,
            flush=True,
        )
        time.sleep(wait)

    def get(self, url, params):
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if self.delay:
                time.sleep(self.delay)

            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=TIMEOUT,
                )
                self.requests += 1
            except requests.RequestException as exc:
                if attempt == MAX_ATTEMPTS:
                    raise

                wait = min(300, 5 * (2 ** (attempt - 1)))
                print(
                    f"\nNETWORK ERROR: {exc}"
                    f"\nretry {attempt}/{MAX_ATTEMPTS} "
                    f"after {duration(wait)}",
                    flush=True,
                )
                time.sleep(wait)
                continue

            self.maybe_wait_primary(response)

            if response.status_code in (403, 429):
                wait = retry_after(response)
                if wait is None:
                    wait = min(300, 60 * (2 ** (attempt - 1)))

                print(
                    "\n" + "!" * 68
                    + f"\nGITHUB RATE LIMIT: HTTP {response.status_code}"
                    + f"\nattempt: {attempt}/{MAX_ATTEMPTS}"
                    + f"\nwaiting: {duration(wait)}"
                    + "\n" + "!" * 68,
                    flush=True,
                )

                if attempt == MAX_ATTEMPTS:
                    return response

                time.sleep(wait)
                continue

            if 500 <= response.status_code <= 599:
                if attempt == MAX_ATTEMPTS:
                    return response

                wait = min(300, 5 * (2 ** (attempt - 1)))
                print(
                    f"\nGITHUB SERVER ERROR HTTP {response.status_code}"
                    f"\nretry after {duration(wait)}",
                    flush=True,
                )
                time.sleep(wait)
                continue

            return response

        raise RuntimeError("request loop ended unexpectedly")


def result_row(row, status, before, client, **kwargs):
    out = {
        "repo_id": int(row.repo_id),
        "cohort": str(row.cohort),
        "full_name": str(row.full_name),
        "default_branch": str(row.default_branch),
        "history_status": status,
        "commit_count": None,
        "first_commit_date": None,
        "last_commit_date": None,
        "development_span_days": None,
        "is_single_commit": None,
        "api_requests": client.requests - before,
        "http_status": None,
        "error_message": "",
        "collected_at": now_iso(),
    }
    out.update(kwargs)
    return out


def collect(client, row):
    before = client.requests
    url = f"{API}/repos/{row.full_name}/commits"

    try:
        newest = client.get(
            url,
            {
                "sha": row.default_branch,
                "per_page": 1,
                "page": 1,
            },
        )
    except requests.RequestException as exc:
        return result_row(
            row,
            "transient_error",
            before,
            client,
            error_message=str(exc),
        )

    status = newest.status_code

    if status == 409:
        return result_row(
            row,
            "empty_repository",
            before,
            client,
            commit_count=0,
            is_single_commit=0,
            http_status=status,
        )

    if status == 404:
        try:
            msg = str(newest.json().get("message", ""))
        except Exception:
            msg = newest.text[:500]

        kind = (
            "branch_not_found"
            if "No commit found for the ref" in msg
            else "not_found"
        )
        return result_row(
            row,
            kind,
            before,
            client,
            http_status=status,
            error_message=msg,
        )

    if status in (403, 429) or 500 <= status <= 599:
        return result_row(
            row,
            "transient_error",
            before,
            client,
            http_status=status,
            error_message=newest.text[:1000],
        )

    if status != 200:
        return result_row(
            row,
            "api_error",
            before,
            client,
            http_status=status,
            error_message=newest.text[:1000],
        )

    try:
        payload = newest.json()
    except ValueError as exc:
        return result_row(
            row,
            "api_error",
            before,
            client,
            http_status=status,
            error_message=f"invalid JSON: {exc}",
        )

    if not payload:
        return result_row(
            row,
            "empty_repository",
            before,
            client,
            commit_count=0,
            is_single_commit=0,
            http_status=status,
        )

    newest_date = commit_date(payload[0])
    count = last_page(newest.headers.get("Link"))

    # With per_page=1, absence of pagination means exactly one commit.
    if count is None:
        return result_row(
            row,
            "single_commit",
            before,
            client,
            commit_count=1,
            first_commit_date=newest_date,
            last_commit_date=newest_date,
            development_span_days=0.0,
            is_single_commit=1,
            http_status=status,
        )

    oldest = client.get(
        url,
        {
            "sha": row.default_branch,
            "per_page": 1,
            "page": count,
        },
    )

    if oldest.status_code != 200:
        kind = (
            "transient_error"
            if (
                oldest.status_code in (403, 429)
                or 500 <= oldest.status_code <= 599
            )
            else "api_error"
        )
        return result_row(
            row,
            kind,
            before,
            client,
            commit_count=count,
            last_commit_date=newest_date,
            is_single_commit=0,
            http_status=oldest.status_code,
            error_message=oldest.text[:1000],
        )

    try:
        old_payload = oldest.json()
    except ValueError as exc:
        return result_row(
            row,
            "api_error",
            before,
            client,
            commit_count=count,
            last_commit_date=newest_date,
            is_single_commit=0,
            http_status=oldest.status_code,
            error_message=f"invalid JSON: {exc}",
        )

    if not old_payload:
        return result_row(
            row,
            "api_error",
            before,
            client,
            commit_count=count,
            last_commit_date=newest_date,
            is_single_commit=0,
            http_status=oldest.status_code,
            error_message="oldest commit page returned empty list",
        )

    oldest_date = commit_date(old_payload[0])
    old_dt = parse_date(oldest_date)
    new_dt = parse_date(newest_date)

    span = None
    if old_dt and new_dt:
        span = (new_dt - old_dt).total_seconds() / 86400.0

    status_name = "ok"
    message = ""

    if span is not None and span < 0:
        status_name = "date_order_error"
        message = "newest commit date precedes oldest commit date"

    return result_row(
        row,
        status_name,
        before,
        client,
        commit_count=count,
        first_commit_date=oldest_date,
        last_commit_date=newest_date,
        development_span_days=span,
        is_single_commit=0,
        http_status=oldest.status_code,
        error_message=message,
    )


def append(row):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    exists = OUTPUT.exists() and OUTPUT.stat().st_size > 0

    with OUTPUT.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def existing_latest():
    if not OUTPUT.exists():
        return pd.DataFrame(columns=FIELDS)

    try:
        df = pd.read_csv(OUTPUT)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=FIELDS)

    return df.drop_duplicates(
        subset=["repo_id"],
        keep="last",
    )


def compact():
    if not OUTPUT.exists():
        return

    df = pd.read_csv(OUTPUT)
    if df.empty:
        return

    df = (
        df.drop_duplicates(
            subset=["repo_id"],
            keep="last",
        )
        .sort_values(
            ["cohort", "repo_id"],
            kind="stable",
        )
    )

    tmp = OUTPUT.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(OUTPUT)


def summary():
    if not OUTPUT.exists():
        return

    df = pd.read_csv(OUTPUT)
    if df.empty:
        return

    df = df.drop_duplicates(
        subset=["repo_id"],
        keep="last",
    )

    print()
    print("=" * 68)
    print("03b HISTORY SUMMARY")
    print("=" * 68)

    for cohort in COHORTS:
        g = df[df["cohort"] == cohort]
        if g.empty:
            continue

        print()
        print(cohort)
        print(f"processed: {len(g):,}")
        print("status:")
        print(
            g["history_status"]
            .value_counts(dropna=False)
            .to_string()
        )

        ok = g[
            g["history_status"].isin(
                ["ok", "single_commit"]
            )
        ]

        if ok.empty:
            continue

        commits = pd.to_numeric(
            ok["commit_count"],
            errors="coerce",
        ).dropna()

        spans = pd.to_numeric(
            ok["development_span_days"],
            errors="coerce",
        ).dropna()

        singles = pd.to_numeric(
            ok["is_single_commit"],
            errors="coerce",
        ).fillna(0)

        print(
            f"single commit: {int(singles.sum()):,}/{len(ok):,} "
            f"({singles.mean()*100:.2f}%)"
        )

        if not commits.empty:
            print(
                "commit count "
                f"median={commits.median():.0f}, "
                f"p75={commits.quantile(.75):.0f}, "
                f"p95={commits.quantile(.95):.0f}"
            )

        if not spans.empty:
            print(
                "development span "
                f"median={spans.median():.2f}d, "
                f"p75={spans.quantile(.75):.2f}d, "
                f"p95={spans.quantile(.95):.2f}d"
            )

    print()
    print(f"Output: {OUTPUT}")


def args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--pilot",
        type=int,
        default=None,
        metavar="N",
        help="process at most N repositories per cohort",
    )

    p.add_argument(
        "--cohort",
        choices=COHORTS,
        default=None,
    )

    p.add_argument(
        "--retry-errors",
        action="store_true",
        help="retry non-terminal rows already present in the output",
    )

    p.add_argument(
        "--request-delay",
        type=float,
        default=REQUEST_DELAY,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20260825,
    )

    return p.parse_args()


def main():
    a = args()

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)

    required = {
        "repo_id",
        "cohort",
        "full_name",
        "default_branch",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"missing required columns: {sorted(missing)}"
        )

    df = df[df["cohort"].isin(COHORTS)].copy()

    if a.cohort:
        df = df[df["cohort"] == a.cohort].copy()

    if "clone_status" in df.columns:
        df = df[df["clone_status"] == "accepted"].copy()

    df = df[
        df["full_name"].notna()
        & df["default_branch"].notna()
    ].copy()

    df["repo_id"] = df["repo_id"].astype(int)

    prev = existing_latest()

    if not prev.empty:
        if a.retry_errors:
            skip = set(
                prev.loc[
                    prev["history_status"].isin(TERMINAL),
                    "repo_id",
                ].astype(int)
            )
        else:
            skip = set(prev["repo_id"].astype(int))

        df = df[~df["repo_id"].isin(skip)].copy()

    if a.pilot is not None:
        if a.pilot <= 0:
            raise ValueError("--pilot must be > 0")

        groups = []

        for cohort in COHORTS:
            g = df[df["cohort"] == cohort]

            if len(g):
                groups.append(
                    g.sample(
                        n=min(a.pilot, len(g)),
                        random_state=a.seed,
                    )
                )

        df = (
            pd.concat(groups, ignore_index=True)
            if groups
            else df.iloc[:0].copy()
        )

    df["cohort"] = pd.Categorical(
        df["cohort"],
        categories=COHORTS,
        ordered=True,
    )

    df = (
        df.sort_values(
            ["cohort", "repo_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    auth = token()

    if not auth:
        print(
            "GitHub token not found.\n"
            "Use one of:\n"
            "  export GITHUB_TOKEN=...\n"
            "  export GH_TOKEN=...\n"
            "or run:\n"
            "  gh auth login",
            file=sys.stderr,
        )
        sys.exit(2)

    gh = GitHub(auth, a.request_delay)

    total = len(df)

    print("=" * 68)
    print("03b REPOSITORY HISTORY COLLECTION")
    print("=" * 68)
    print(f"input:          {INPUT}")
    print(f"output:         {OUTPUT}")
    print(f"repositories:   {total:,}")
    print(f"pilot:          {a.pilot if a.pilot else 'no'}")
    print(f"retry errors:   {a.retry_errors}")
    print("scope:          default branch only")
    print(
        "metrics:        commit_count, first/last commit date, "
        "development_span_days"
    )
    print()

    if total == 0:
        print("Nothing to process.")
        compact()
        summary()
        return

    started = time.time()

    for i, row in enumerate(
        df.itertuples(index=False),
        start=1,
    ):
        r = collect(gh, row)
        append(r)

        elapsed = time.time() - started
        speed = i / elapsed if elapsed else 0.0
        eta = (total - i) / speed if speed else 0.0

        span = r["development_span_days"]
        span_text = (
            "-"
            if span is None
            else f"{span:.2f}"
        )

        print(
            f"[{i:>6,d}/{total:,}] "
            f"{r['cohort']:<12} "
            f"{r['full_name'][:45]:<45} "
            f"{r['history_status']:<18} "
            f"commits={str(r['commit_count']):>7} "
            f"span={span_text:>10}d "
            f"req={r['api_requests']} "
            f"elapsed={duration(elapsed)} "
            f"ETA={duration(eta)}",
            flush=True,
        )

    compact()

    elapsed = time.time() - started

    print()
    print("=" * 68)
    print("03b FINISHED")
    print("=" * 68)
    print(f"processed:      {total:,}")
    print(f"API requests:   {gh.requests:,}")
    print(f"requests/repo:  {gh.requests / total:.3f}")
    print(f"elapsed:        {duration(elapsed)}")

    summary()


if __name__ == "__main__":
    main()
