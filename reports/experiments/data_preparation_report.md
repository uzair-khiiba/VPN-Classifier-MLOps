# Phase 2: Data preparation report

**Commands:** `python -m src.prepare_data` (the project's pipeline, unmodified), then `python -m experiments.verify_prepared_data` (independent checks).

**Result:** preparation succeeded and all checks passed. Raw numbers are in [prepared_data_checks.json](prepared_data_checks.json).

## What one row represents

After preparation, **one row is one attacker IP**, summarising everything that IP did during the month:

| Before | After |
|---|---|
| 61,629,685 rows, one per attack signal | 148,259 rows, one per IP |
| Row example: "IP 123 attacked watcher 45 with `http:scan` at 14:02" | Row example: "IP 123 sent 53 signals over 9 days, and shares its HTTP header hash with 690 other IPs" |

The `n_signals` column sums to exactly 61,629,685, so no signal was lost during aggregation.

## Why aggregation is necessary

The label describes an IP ("this IP is a VPN/proxy"), not an individual attack. Training on signals directly would cause two problems:

1. **Busy IPs would dominate.** One IP sent 1,466,169 signals while the median IP sent 37.
2. **The same IP would leak across the split.** A random split of signals would put the same IP in both train and test, so the model could score well by recognising IPs instead of learning behaviour.

## What each split contains

| Split | IPs | VPN/proxy | Other | Positive rate |
|---|---:|---:|---:|---:|
| Train | 118,607 | 2,312 | 116,295 | 1.9493% |
| Test | 29,652 | 578 | 29,074 | 1.9493% |
| Total | 148,259 | 2,890 | 145,369 | 1.9493% |

The code implements a stratified 80/20 split, as documented. "Stratified" means both sets keep the same share of positives, which is why the rate matches. A hash of the test IP list (`7a8a6674be67b514`) is recorded so later work can prove that the split never changed.

## Why splitting by IP matters

Each IP appears exactly once, with 0 duplicates and 0 IPs shared between train and test. The test score therefore measures performance on IPs the model has never seen.

## How the imbalance looks

About 1 IP in 51 is positive. A model that always answers "not VPN" would score **98.05% accuracy** and catch none of them, which is why accuracy is not the main metric. With only 578 positives in the test set, one positive is worth about 0.17 percentage points of recall, so small F1 differences can be noise.

## Verification

| Check | Result |
|---|---|
| About 148,259 IPs | Exactly 148,259 |
| One record per IP | 0 duplicates |
| No IP in more than one split | 0 overlap |
| Expected features and target | 28 features plus `label` and `attacker_ip_enum`, no missing or unexpected columns |
| Missing values in features | None |
| Negative values | None |
| Constant features | None |
| IP ID or label used as a feature | No |

## Data-quality notes

- `watcher_country` is null in 349,386 signals, so `n_watcher_countries` can be 0. The API allows this.
- `attacker_as_num` is null in 462 signals, so an IP with no known ASN gets `asn_unique_ips = 0`.
- 70 IPs appear under more than one attacker ASN, and `asn_unique_ips` takes the largest ASN.
- In the Shodan file, 150,083 of 197,679 cells are the literal text `{}`. Parsing turns exactly those cells into an empty result, so no cell failed to parse.
- `median_gap_s` is 0 for the median IP in both classes, because many signals share a timestamp.

## Leakage check

Leakage means the model learns from information it would not legitimately have. Three kinds were checked.

### 1. Does any feature directly reveal the label?

**No evidence of this.** Each feature was scored alone on training data. A label proxy would rank positives almost perfectly (AUC near 1.0), but the strongest features are well short of that:

| Feature | AUC alone | VPN median | Other median |
|---|---:|---:|---:|
| `n_tcp_ports` | 0.900 | 2 | 0 |
| `headers_hash_shared` | 0.880 | 690 | 0 |
| `has_shodan` | 0.877 | 1 | 0 |
| `n_active_days` | 0.741 | 9 | 3 |
| `share_http_spam` | 0.723 | 0.41 | 0.27 |

The label itself is excluded from feature calculations. It appears only as a separate output column.

### 2. Do features use test-set information?

**Yes, in a limited way, and this is the main finding.** Four features count across all IPs, including those in our test set:

| Feature | Train rows that change if test IPs are removed | Typical value compared with now |
|---|---:|---:|
| `headers_hash_shared` | 13.8% | About 60% |
| `jarm_shared` | 8.1% | About 61% |
| `ja3s_shared` | 8.4% | About 61% |
| `asn_unique_ips` | 93.2% | About 80% |

The fingerprint counts also include Kaggle's 49,420 unlabelled test IPs, which is why the typical value drops to about 60%: train IPs are 118,607 of the 197,679 IPs in the Shodan file.

**This is not label leakage**, because these counts use no labels. It is *transductive* feature engineering: the feature table was built with knowledge of which other IPs exist in the test period. It does create two real concerns:

- The test score assumes that these population counts are available when predicting.
- A real deployment needs a reference table of fingerprint and ASN counts, and its counts must be on the same scale as training. The API takes these values as inputs and doesn't explain where they come from.

Experiment D measures how much this affects the score.

### 3. Is Shodan data available at prediction time, and does it define the label?

Not fully answerable from this data:

- **Row presence is not a leak.** All 148,259 IPs have a Shodan row, in both classes.
- **Having ports is strongly associated.** 98.0% of VPN IPs have open ports recorded, against 22.7% of others. IPs with Shodan data are about 160 times more likely to be VPN/proxy (7.91% vs 0.05%).
- **What we cannot tell** is how CrowdSec created the labels, and whether Shodan-style scans played any part. Neither file records this, and it should be raised with the instructor or checked against the competition description.
- **Timing is unknown.** The file has no timestamps, so we can't confirm the Shodan snapshot predates the attacks.

## Conclusion

The prepared data is correct and consistent with the documentation, so it is safe to start the baseline. The fixed split will not change. The transductive features stay in the baseline for reproducibility, and are measured separately in Experiment D.
