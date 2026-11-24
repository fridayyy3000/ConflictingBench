# ConflictBench comparison

- `nvidia_baseline`: conflictbench_all_nvidia_baseline_details_20261113_151422.csv
- `nvidia_govrag`: conflictbench_all_nvidia_govrag_details_20261113_145408.csv

## Overall

| metric                | nvidia_baseline | nvidia_govrag |
|-----------------------|-----------------|---------------|
| answer correct        | 17/30 (57%)     | 29/30 (97%)   |
| gold source selected  | 17/30 (57%)     | 28/30 (93%)   |
| gold in final context | 21/30 (70%)     | 30/30 (100%)  |
| mean latency (s)      | 26.9            | 38.7          |

## Decisions

| decision              | nvidia_baseline | nvidia_govrag |
|-----------------------|-----------------|---------------|
| answered              | 17              | 28            |
| insufficient_evidence | 11              | 2             |
| unresolved_conflict   | 2               | 0             |

## Selected document role

| role | nvidia_baseline | nvidia_govrag |
|------|-----------------|---------------|
| gold | 17              | 28            |
| none | 13              | 2             |

## Answer accuracy by answer type

| answer type       | nvidia_baseline | nvidia_govrag |
|-------------------|-----------------|---------------|
| boolean (n=3)     | 3/3 (100%)      | 3/3 (100%)    |
| categorical (n=3) | 3/3 (100%)      | 3/3 (100%)    |
| date (n=3)        | 2/3 (67%)       | 3/3 (100%)    |
| entity (n=3)      | 3/3 (100%)      | 3/3 (100%)    |
| numeric (n=18)    | 6/18 (33%)      | 17/18 (94%)   |

## Per question

| qid  | type        | nvidia_baseline answer | nvidia_govrag answer | nvidia_baseline source       | nvidia_govrag source         |
|------|-------------|------------------------|----------------------|------------------------------|------------------------------|
| Q001 | numeric     | OK                     | OK                   | OK                           | OK                           |
| Q002 | numeric     | miss                   | OK                   | miss [insufficient_evidence] | OK                           |
| Q003 | numeric     | miss                   | OK                   | miss [insufficient_evidence] | OK                           |
| Q004 | numeric     | OK                     | OK                   | OK                           | OK                           |
| Q005 | numeric     | miss                   | OK                   | miss [insufficient_evidence] | OK                           |
| Q006 | numeric     | miss                   | OK                   | miss [insufficient_evidence] | OK                           |
| Q007 | numeric     | miss                   | OK                   | miss [unresolved_conflict]   | OK                           |
| Q008 | numeric     | miss                   | OK                   | miss [insufficient_evidence] | OK                           |
| Q009 | numeric     | miss                   | OK                   | miss [insufficient_evidence] | OK                           |
| Q010 | numeric     | miss                   | OK                   | miss [insufficient_evidence] | OK                           |
| Q011 | numeric     | OK                     | OK                   | OK                           | OK                           |
| Q012 | numeric     | OK                     | OK                   | OK                           | OK                           |
| Q013 | numeric     | miss                   | OK                   | miss [insufficient_evidence] | OK                           |
| Q014 | numeric     | miss                   | OK                   | miss [unresolved_conflict]   | OK                           |
| Q015 | numeric     | miss                   | OK                   | miss [insufficient_evidence] | OK                           |
| Q016 | numeric     | OK                     | miss                 | OK                           | miss [insufficient_evidence] |
| Q017 | numeric     | miss                   | OK                   | miss [insufficient_evidence] | OK                           |
| Q018 | numeric     | OK                     | OK                   | OK                           | OK                           |
| Q019 | date        | OK                     | OK                   | OK                           | OK                           |
| Q020 | date        | miss                   | OK                   | miss [insufficient_evidence] | OK                           |
| Q021 | date        | OK                     | OK                   | OK                           | OK                           |
| Q022 | entity      | OK                     | OK                   | OK                           | OK                           |
| Q023 | entity      | OK                     | OK                   | OK                           | OK                           |
| Q024 | entity      | OK                     | OK                   | OK                           | OK                           |
| Q025 | categorical | OK                     | OK                   | OK                           | OK                           |
| Q026 | categorical | OK                     | OK                   | OK                           | OK                           |
| Q027 | categorical | OK                     | OK                   | OK                           | OK                           |
| Q028 | boolean     | OK                     | OK                   | OK                           | OK                           |
| Q029 | boolean     | OK                     | OK                   | OK                           | OK                           |
| Q030 | boolean     | OK                     | OK                   | OK                           | miss [insufficient_evidence] |

