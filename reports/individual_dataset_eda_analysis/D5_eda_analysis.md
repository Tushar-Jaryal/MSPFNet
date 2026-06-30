# D5 EDA Analysis - Mixed Bearing-Gearbox Compound Faults

## Dataset Structure

D5 contains 64 files, three channels, and 15 fine-grained classes. Most classes have four files, while `NOR` has eight. The labels include bearing faults, gearbox faults, mixed faults, and normal operation. This makes D5 a small-sample, high-cardinality compound-fault dataset.

## Signal Behavior

The dataset has wide feature variation, but the variation does not cleanly separate all labels. `NOR` has the highest RMS at 0.335, which is counterintuitive and shows that healthy recordings are not simply low-energy. `BRG_BF`, `BRG_IO`, and `BRG_OR` are also high-energy. `BRG_IR` has very high kurtosis at 16.995 and crest factor 13.742 despite moderate RMS.

Several mixed classes sit in the middle of the feature space, which makes them difficult to separate from their single-component relatives. The closest pairs include `MIX_II` versus `MIX_OI` at 0.727, `GBX_EG` versus `GBX_WR` at 0.852, `GBX_WR` versus `MIX_OS` at 0.908, `MIX_IS` versus `MIX_OS` at 0.975, and `BRG_CF` versus `BRG_OR` at 1.051.

## Temporal Scale

The period analysis recommends a 256-sample window, with median period 15 samples and 75th percentile period 28 samples. This reflects short impulse-like events. However, the maximum selected period reaches 3191 samples, so some files still contain longer modulation structure. D5 likely needs both short impulse detection and broader context.

## Modeling Implication

D5 is difficult because related single and compound classes overlap. Models need to represent component interactions, not only detect fault presence. The poor MSPF-Net results on D5 indicate that the current branch fusion and periodic feature design do not yet handle this compound taxonomy well.

## Thesis Takeaway

D5 is a compact but difficult compound-fault dataset. It is not governed by simple healthy/fault energy separation because normal has the highest RMS. The main challenge is overlap between related mixed and single-component classes, making D5 the clearest test of compound-fault representation quality.
