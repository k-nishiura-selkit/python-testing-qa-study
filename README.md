# Replication Package

This repository contains the replication package for the paper:

**Testing and Quality Assurance in Small Python Projects Before and
After the Rise of Generative AI**

The study analyzes testing and quality-assurance practices in 20,000
small Python GitHub repositories: 10,000 Pre-AI repositories and 10,000
Contemporary repositories.

## Repository Selection and Data

The repository selection process is implemented in
`01_collect_candidates.py`--`03_clone_repositories.py`.

`config.yaml` contains the parameters, thresholds, and keyword lists
used for repository collection and filtering. These provide additional
details on the repository selection criteria that are omitted from the
paper due to space limitations.

Intermediate CSV files are included to make the repository selection
process traceable. They record the intermediate and final results of
repository filtering, acquisition, and selection.

The processed datasets used for the analyses are provided under
`data/processed/`. The cloned GitHub repositories themselves are not
included because of their size.

## Scripts

The complete pipeline consists of:

``` text
01_collect_candidates.py
02_filter_candidates.py
03_clone_repositories.py
04_collect_repository_history.py
05_analyze_test_infrastructure.py
06_analyze_quality_metrics.py
07_analyze_results.py
```

The scripts are intended to be executed in numerical order.

To reproduce the results reported in the paper from the provided
processed data, run:

``` bash
python 07_analyze_results.py
```

The generated results are stored under `data/processed/07_results/`.

Runtime files such as cloned repositories, logs, and SQLite checkpoint
databases are not included.
