# Benchmark Dataset EDA Analysis

This report summarizes the current EDA evidence for the active seven-dataset benchmark:
D1, D2, D3C1, D3C9, D4, D5, and D8. Dataset 1 uses the new individual EDA
report under `reports/dataset_1_EDA_report`; the remaining datasets use the
shared EDA tables in `results/tables`, processed metadata, and the latest
aggregate benchmark table.

## D1 - BearingTV

D1 is structurally clean and balanced. It contains 60 files, five fault classes
(`BRG_BF`, `BRG_CF`, `BRG_IR`, `BRG_OR`, `NOR`), and 12 files per class. The
individual D1 report shows two recorded channels per raw file, each with
2,000,000 samples and no missing or non-finite values. The condition grid is
also balanced: each fault has three trials for each condition A-D.

The most important signal finding is that Channel 1 carries the diagnostic
vibration information, while Channel 2 behaves like a switching or reference
signal. The mean Channel 1-2 correlation is only 0.000097, and Channel 2 has
near-constant switching-like statistics, including kurtosis around -1.987. It
should not be interpreted as a second independent vibration sensor without care.

Channel 1 provides strong class separability. `BRG_CF` is the most impulsive
class, with mean kurtosis 62.1, mean crest factor 31.0, and peak-to-peak
amplitude 1.71. `BRG_IR` is the highest-energy class, with mean RMS 0.0608 in
the individual report and very high peak-to-peak amplitude. `NOR`, `BRG_OR`,
and `BRG_BF` are lower-energy and closer together; the closest pair in the
shared EDA separation table is `BRG_BF` versus `BRG_OR` with distance 0.507.

The main caveat is that several `BRG_CF` files are marked as IQR outliers
because their impulsive kurtosis is class-defining, not necessarily because the
files are bad. `I-A-1.mat` is more concerning as an energy outlier because its
RMS is much higher than typical inner-race files. D1 is therefore a clean and
mostly easy bearing benchmark, useful as a pipeline sanity check and as a test
of condition-holdout generalization rather than as the hardest discrimination
case.

Thesis takeaway: D1 is a balanced, clean, single-vibration-channel bearing
dataset with strong statistical class separation. Combined faults are dominated
by impulsiveness, inner-race faults by elevated energy, and healthy/outer-race
classes remain low-energy. This explains the high D1 model performance.

## D2 - Planetary Gearbox

D2 contains 80 files, five classes, and four channels. Each class has 16 files,
so the file-level class distribution is balanced. The classes are `GBX_BT`
(broken tooth), `GBX_MT` (missing tooth), `GBX_RC` (root crack), `GBX_WR`
(wear), and `NOR`. The signals are long: class-level durations are about
301-318 seconds, much longer than D1.

D2 is a strong architectural discriminator because the fault classes differ in
both energy and impulsiveness. `GBX_WR` has the largest RMS at 81.94, while
`GBX_MT` has the strongest impulsive behavior with kurtosis 47.81 and the
largest peak-to-peak amplitude at 2535.10. `GBX_RC` is comparatively subtle,
with RMS 11.79 and kurtosis 3.08, close to ordinary vibration statistics.
`NOR` is not the lowest-energy class; its RMS is 19.06 and kurtosis is 11.95,
so simple normal-versus-fault thresholding is not sufficient.

Pairwise separation is moderate rather than extreme. The closest pair is
`GBX_BT` versus `NOR` with distance 1.976, followed by `GBX_BT` versus
`GBX_RC` and `GBX_BT` versus `GBX_MT`. The farthest pair is `GBX_MT` versus
`GBX_RC` at 3.928. This means the dataset has recognizable statistical
structure, but several classes overlap enough that multi-scale temporal and
spectral modeling matters.

The period analysis recommends an 8192-sample window, with median period
1257.5 samples and 75th percentile period 3006.75 samples. That is consistent
with D2's planetary gearbox dynamics: long windows are needed to capture slow
planet-carrier and modulation patterns. The best results are achieved by
InceptionTime and MixNet, both above 81 percent window macro-F1, supporting the
need for multi-scale temporal features.

Thesis takeaway: D2 is balanced but physically complex. Missing tooth and wear
faults are high-energy or highly impulsive, while root crack is subtle and
normal is not trivially low-energy. Long-window multi-scale models are favored,
making D2 one of the benchmark's most useful architecture-discrimination
datasets.

## D3C1 - CWRU Bearing Subset

D3C1 contains 40 single-channel files from the CWRU bearing subset. It has four
classes: `BRG_BF`, `BRG_IR`, `BRG_OR`, and `NOR`. The fault classes each have
12 files, while `NOR` has 4 files, so the class distribution is less balanced
than D1. The dataset is still small and clean enough that most strong models
solve it.

The class statistics are well separated. `BRG_OR` has the highest RMS at 0.670
and the largest peak-to-peak amplitude at 7.039. `BRG_IR` is next with RMS
0.292 and high spectral entropy, while `BRG_BF` is weaker at RMS 0.139.
`NOR` is clearly low energy, with RMS 0.074 and peak-to-peak amplitude 0.598.
This monotonic energy separation makes the task easier than D4 or D3C9.

Pairwise separation confirms this. The closest pair is `BRG_BF` versus `NOR`
with distance 2.001, and the farthest pair is `BRG_IR` versus `NOR` at 4.048.
Even the closest pair is more separated than the hardest pairs in D1, D3C9, or
D4. The period analysis recommends an 8192-sample window because dominant
periods are long, with median period about 2003.5 samples.

The caveat is that D3C1 can overstate model strength. It is a small,
well-structured bearing dataset with clean class separation, and several models
reach 100 percent file-level macro-F1. It is useful for checking whether the
pipeline and models can learn bearing signatures under load holdout, but it is
not a strong stress test for compound or cross-component diagnostics.

Thesis takeaway: D3C1 is nearly solved because bearing fault classes separate
strongly in energy and amplitude. It validates pipeline correctness and
bearing-fault learning, but it should not be used alone to claim broad
generalization.

## D3C9 - Gearbox Multichannel Subset

D3C9 is the hardest dataset in the benchmark. It contains 20 files, nine
channels, and nine classes: five bearing/normal classes plus four gearbox
classes. Most classes have only two files, while `NOR` has four. This creates a
severe sample-size limitation before modeling even begins.

The statistical profile is irregular and overlapping. `BRG_BF` has extremely
high kurtosis, 389.95, and crest factor, 141.90, but very low RMS, 0.007.
`GBX_WR` and `BRG_IR` also have low RMS but high crest factors above 80.
`BRG_CF` and `NOR` have similar high RMS values, 0.362 and 0.339, and they are
the closest pair in the separation table with distance 0.365. `BRG_OR` and
`GBX_BT` are also very close, distance 0.508, and `GBX_MT` and `GBX_RC` are
close at 0.639.

The channel and period behavior is also difficult. The period table recommends
a short 256-sample window because the median selected period is only 5 samples,
but the maximum period reaches 3080 samples. This means different channels and
faults carry inconsistent temporal scales. Some channels appear to be dominated
by very short-period or nearly discrete patterns, while others show much longer
structure. A single fixed representation is unlikely to suit all classes.

The modeling results align with the EDA: even the best current D3C9 runs are
below 30 percent window macro-F1. The low file count, nine-class label space,
multi-channel heterogeneity, and near-overlapping class pairs make D3C9 a
data-limited benchmark as much as a model-limited one.

Thesis takeaway: D3C9 should be reported with explicit caveats. Its poor
performance reflects sparse files per class, many channels, heterogeneous
temporal scales, and low class separability. It is valuable as a stress test,
but low accuracy should not be interpreted as model failure alone.

## D4 - Multi-Mode Gearbox

D4 contains 240 files and eight gear-related classes. The file counts are
imbalanced: `GBX_MT` and `NOR` have 12 files each, while most other classes have
36. The catalog reports one processed channel for the benchmark, and all classes
have the same analyzed duration of 76.8 seconds in the shared EDA statistics.

D4 is difficult because the class-level statistics are nearly indistinguishable.
RMS values lie in a narrow range from 1.212 to 1.352, peak-to-peak amplitude is
almost constant around 4.31-4.33, crest factor stays near 3, and kurtosis ranges
only from 2.30 to 4.01. Even normal has one of the highest RMS values, so
amplitude does not separate healthy from faulty operation.

Pairwise separation confirms the overlap. The closest pairs are `GBX_BT`
versus `GBX_WR` at 0.281, `MIX_TI` versus `MIX_TO` at 0.387, `GBX_MT` versus
`GBX_PT` at 0.591, and `GBX_CK` versus `NOR` at 0.647. These are exactly the
kinds of pairs that simple statistical features and ordinary CNNs confuse. The
best separations involve normal against some tooth-related classes, but much of
the label space remains compact.

The period analysis recommends a 256-sample window, with median period 21.5
samples but a maximum period 4352 samples. This suggests short-window local
patterns are important, while some low-frequency or long-period structure may
also exist. The best D4 result comes from MSPF-Net with RF, which supports the
hypothesis that learned periodic features plus a non-neural decision boundary
can help on subtle gear discrimination.

Thesis takeaway: D4 is not hard because of missing data; it is hard because the
classes are genuinely close in the feature space. Its near-identical RMS,
peak-to-peak, and kurtosis profiles make it the strongest test of whether a
model can learn subtle periodic and spectral gearbox differences.

## D5 - Mixed Bearing-Gearbox

D5 contains 64 files, three channels, and 15 fine-grained classes. Most classes
have four files, while `NOR` has eight. This makes the task small-sample and
high-cardinality: many classes must be learned from very few recordings.

The dataset combines bearing faults, gearbox faults, mixed faults, and normal
operation. Class statistics vary widely, but not in a way that cleanly separates
all labels. `NOR` has the highest RMS at 0.335, which is counterintuitive and
shows that healthy recordings are not simply low-energy. `BRG_BF`, `BRG_IO`,
and `BRG_OR` are also high-energy. `BRG_IR` has very high kurtosis, 16.995, and
high crest factor, 13.742, despite moderate RMS. Several mixed classes occupy
the middle of the feature space.

Pairwise separation shows why D5 is difficult. The closest pairs include
`MIX_II` versus `MIX_OI` at 0.727, `GBX_EG` versus `GBX_WR` at 0.852,
`GBX_WR` versus `MIX_OS` at 0.908, `MIX_IS` versus `MIX_OS` at 0.975, and
`BRG_CF` versus `BRG_OR` at 1.051. These pairs are semantically related or
compound-overlap cases, where one component fault can mask or resemble another.

The period analysis recommends a 256-sample window, with median period 15
samples and 75th percentile 28 samples. The recommended short window reflects
fast local impulses, but the maximum selected period reaches 3191 samples, so
some files still contain longer modulation structure. Models that can combine
short impulse detection with broader context are favored. Recommended-fine
InceptionTime, ResNet1D, MixNet, and SE-CNN1D perform best, while current
MSPF-Net variants struggle on the full compound taxonomy.

Thesis takeaway: D5 is a compact but difficult compound-fault dataset. It is
not dominated by simple healthy/fault energy separation, because normal has the
highest RMS. The main challenge is overlap between related mixed and
single-component classes, making D5 the clearest test of compound-fault
representation quality.

## D8 - HUST Gearbox

D8 contains 90 files, four channels, and three balanced classes:
`GBX_BT`, `GBX_MT`, and `NOR`, with 30 files per class. It is structurally much
cleaner than D3C9 and has fewer classes than D4 or D5.

The key signal finding is that `GBX_MT` is clearly separable. It has RMS 0.113,
peak-to-peak amplitude 3.092, crest factor 14.31, and kurtosis 44.84. By
contrast, `GBX_BT` and `NOR` are much closer: RMS values are 0.039 and 0.037,
and their pairwise separation is only 1.049. `GBX_MT` is far from both other
classes, with distances around 3.80.

The period analysis recommends an 8192-sample window, with median period
2605.5 samples and 75th percentile 3908.25 samples. This is a long-window,
multi-channel gearbox problem. Models need enough context to capture repeated
gearbox structure rather than only short impulses.

The model results match the EDA. MixNet nearly solves D8 in fixed-fine mode,
with 99.93 percent window macro-F1 and 100 percent file macro-F1. Several other
models exceed 90 percent in recommended or fixed settings. The remaining
difficulty is mostly distinguishing `GBX_BT` from `NOR`; missing tooth is much
easier.

Thesis takeaway: D8 is a clean, balanced, long-window gearbox dataset. Missing
tooth is strongly separable, while broken tooth is closer to normal. The
dataset rewards multi-channel, long-context modeling and is useful for testing
long-window architectures rather than sparse-class robustness.

## Cross-Dataset Summary

The seven datasets span different difficulty modes:

| Dataset | Main Challenge | EDA Difficulty Signal | Modeling Implication |
|---|---|---|---|
| D1 | Condition holdout, mild class overlap | Strong Channel 1 separation; balanced files | Pipeline sanity check; high expected scores |
| D2 | Planetary gearbox modulation | Long periods and overlapping gear-fault pairs | Needs long-window multi-scale modeling |
| D3C1 | Small but clean CWRU bearing task | Clear energy separation | Mostly solved; validates bearing learning |
| D3C9 | Sparse nine-class multichannel task | Very close class pairs and only 2-4 files per class | Data-limited stress test |
| D4 | Subtle gearbox modes | Near-identical RMS/peak/kurtosis profiles | Requires periodic/spectral discrimination |
| D5 | Compound mixed faults | Many related classes with small file counts | Tests compound-fault representation |
| D8 | Long-window gearbox classification | Missing tooth easy; broken tooth close to normal | Rewards long-context multichannel models |

Overall, D1 and D3C1 are the easiest and primarily validate the processing and
modeling pipeline. D8 is also clean but tests long-window multichannel gearbox
modeling. D2 is a strong architecture discriminator because planetary gearbox
dynamics require long-range multi-scale features. D4 and D5 are the most
important thesis datasets for subtle and compound faults. D3C9 should be treated
as a severe sparse-data stress test with explicit reporting caveats.
