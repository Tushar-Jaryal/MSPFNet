# D8 EDA Analysis - HUST Gearbox

## Dataset Structure

D8 is a clean, balanced gearbox dataset with 90 files, four channels, and three classes: `GBX_BT`, `GBX_MT`, and `NOR`. Each class has 30 files. Compared with D3C9 and D5, D8 has fewer classes and a much healthier file count per class.

## Signal Behavior

The key signal finding is that `GBX_MT` is clearly separable. It has RMS 0.113, peak-to-peak amplitude 3.092, crest factor 14.31, and kurtosis 44.84. By contrast, `GBX_BT` and `NOR` are much closer: their RMS values are 0.039 and 0.037, and their pairwise separation is only 1.049. `GBX_MT` is far from both other classes, with pairwise distances around 3.80.

This means D8 is not uniformly hard. Missing tooth is the easy class; the remaining challenge is distinguishing broken tooth from normal operation.

## Temporal Scale

The period analysis recommends an 8192-sample window. The median selected period is 2605.5 samples and the 75th percentile is 3908.25 samples, indicating that D8 is a long-context, multi-channel gearbox dataset. Models need to capture repeated gearbox structure rather than only short impulses.

## Modeling Implication

D8 rewards multi-channel, long-window architectures. MixNet nearly solves the fixed-fine D8 benchmark, reaching 99.9 percent window macro-F1 and 100 percent file macro-F1. Strong performance here indicates good long-window gearbox modeling, but not necessarily robustness to sparse or compound labels.

## Thesis Takeaway

D8 is structurally clean and balanced. Missing tooth is strongly separable, while broken tooth is closer to normal. The dataset is useful for testing long-context multi-channel gearbox modeling rather than sparse-class robustness.
