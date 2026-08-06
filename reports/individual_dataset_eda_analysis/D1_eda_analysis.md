# D1 EDA Analysis - BearingTV

## Dataset Structure

D1 is a clean and balanced bearing-fault dataset. The benchmark catalog contains 60 files, five fine-grained classes, and 12 files per class: `BRG_BF`, `BRG_CF`, `BRG_IR`, `BRG_OR`, and `NOR`. The new individual D1 report shows two raw channels per file, each with 2,000,000 samples, no missing values, and no non-finite values. The condition grid is also balanced: each fault has three trials under each condition A-D.

## Signal Behavior

The most important channel-level result is that Channel 1 carries the diagnostic vibration information, while Channel 2 behaves like a switching or reference-like signal. The mean Channel 1-2 correlation is only 0.000097, and Channel 2 has near-constant switching statistics with kurtosis around -1.987. Channel 2 should therefore not be treated as a second independent vibration sensor without care.

Channel 1 separates the classes strongly. `BRG_CF` is the most impulsive class, with mean kurtosis 62.1, crest factor 31.0, and peak-to-peak amplitude 1.71. `BRG_IR` is the highest-energy class, with mean RMS 0.0608 and high peak-to-peak amplitude. `NOR`, `BRG_OR`, and `BRG_BF` are lower-energy and closer together; the closest pair in the shared EDA separation table is `BRG_BF` versus `BRG_OR` with distance 0.507.

## Caveats

The IQR outlier list marks several `BRG_CF` files, but these are not necessarily bad recordings. They are outliers because the combined-fault class is genuinely impulsive. The more important caution is `I-A-1.mat`, which appears as an energy outlier with much higher RMS than other inner-race files.

## Modeling Implication

D1 should be considered an easy-to-moderate benchmark. It is useful for validating preprocessing, label mapping, and condition-holdout evaluation. The main challenge is not class imbalance or data quality; it is whether models generalize from condition A/B training to held-out condition C/D profiles.

## Thesis Takeaway

D1 is a balanced, clean, single-diagnostic-channel bearing dataset with strong statistical class separation. Combined faults are dominated by impulsive behavior, inner-race faults by elevated energy, and healthy/outer-race classes remain low-energy. This explains the high D1 model performance and makes D1 a pipeline sanity-check dataset rather than the hardest benchmark.
