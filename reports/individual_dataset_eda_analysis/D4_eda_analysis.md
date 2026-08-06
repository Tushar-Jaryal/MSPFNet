# D4 EDA Analysis - Multi-Mode Gearbox

## Dataset Structure

D4 contains 240 files and eight gear-related classes: `GBX_BT`, `GBX_CK`, `GBX_MT`, `GBX_PT`, `GBX_WR`, `MIX_TI`, `MIX_TO`, and `NOR`. The class distribution is imbalanced: `GBX_MT` and `NOR` have 12 files each, while most other classes have 36. The benchmark uses one processed channel.

## Signal Behavior

D4 is difficult because the class-level statistics are nearly indistinguishable. RMS values lie in a narrow range from 1.212 to 1.352, peak-to-peak amplitudes stay around 4.31-4.33, crest factor stays near 3, and kurtosis ranges only from 2.30 to 4.01. Even `NOR` has one of the highest RMS values, so normal/fault discrimination cannot rely on simple energy thresholds.

The pairwise separation table confirms this overlap. The closest pairs are `GBX_BT` versus `GBX_WR` at 0.281, `MIX_TI` versus `MIX_TO` at 0.387, `GBX_MT` versus `GBX_PT` at 0.591, and `GBX_CK` versus `NOR` at 0.647. These are precisely the pairs that simple statistical features and generic CNNs are likely to confuse.

## Temporal Scale

The period analysis recommends a 256-sample window, with median period 21.5 samples and a maximum period 4352 samples. This suggests that short local patterns are important, but some longer low-frequency or modulation structure may also exist.

## Modeling Implication

D4 is not hard because of missing data. It is hard because the classes are genuinely close in feature space. Its best current result comes from MSPF-Net with an RF head, which supports the idea that learned periodic features plus a non-neural decision boundary can help on subtle gearbox discrimination.

## Thesis Takeaway

D4 is the benchmark's cleanest example of subtle gearbox-mode difficulty. Its near-identical RMS, peak-to-peak, and kurtosis profiles make it a strong test of whether a model can learn fine periodic and spectral differences rather than relying on amplitude or impulsiveness.
