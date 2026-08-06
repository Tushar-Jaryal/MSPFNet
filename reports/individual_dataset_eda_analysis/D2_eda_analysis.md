# D2 EDA Analysis - Planetary Gearbox

## Dataset Structure

D2 is a balanced planetary gearbox dataset with 80 files, four channels, and five classes: `GBX_BT`, `GBX_MT`, `GBX_RC`, `GBX_WR`, and `NOR`. Each class has 16 files. The recordings are long relative to the other benchmark datasets, with class-level analyzed durations around 301-318 seconds.

## Signal Behavior

D2 is physically complex because the fault classes differ in both energy and impulsiveness. `GBX_WR` has the largest RMS at 81.94, while `GBX_MT` has the strongest impulsive behavior with kurtosis 47.81 and peak-to-peak amplitude 2535.10. `GBX_RC` is much subtler, with RMS 11.79 and kurtosis 3.08, close to ordinary vibration statistics. `NOR` is not trivially low-energy: its RMS is 19.06 and its kurtosis is 11.95.

The pairwise separation table shows moderate class separation rather than a simple easy split. The closest pair is `GBX_BT` versus `NOR` with distance 1.976, followed by `GBX_BT` versus `GBX_RC` and `GBX_BT` versus `GBX_MT`. The farthest pair is `GBX_MT` versus `GBX_RC` at 3.928. This means the dataset contains useful statistical structure, but multiple fault classes overlap enough that amplitude-only methods are insufficient.

## Temporal Scale

The period analysis supports long-window modeling. D2 has a recommended 8192-sample window, with median selected period 1257.5 samples and 75th percentile period 3006.75 samples. This is consistent with planetary gearbox dynamics, where carrier motion and modulation patterns unfold over longer time spans than short bearing impulses.

## Modeling Implication

D2 is one of the strongest architecture-discrimination datasets. Models that can integrate long-range, multi-scale temporal and spectral information perform best. This aligns with the strong performance of InceptionTime and MixNet, while simpler or poorly matched models struggle.

## Thesis Takeaway

D2 is balanced but physically complex. Missing tooth and wear faults create high-energy or highly impulsive signatures, while root crack is subtle and normal is not simply low-energy. D2 rewards long-window multi-scale models and is a key dataset for testing whether an architecture can handle planetary gearbox modulation.
