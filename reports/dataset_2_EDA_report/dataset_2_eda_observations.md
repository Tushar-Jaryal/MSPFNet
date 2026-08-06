# D2 EDA Observations

- D2 contains 80 files, 4 loaded channels per file on average, and 5 fault classes.
- Some long recordings were strided to MAX_SAMPLES_PER_FILE for notebook-level statistics; this preserves broad record coverage while keeping EDA interactive.
- Primary diagnostic channel selected by feature-separation score: Channel_1.
- Fault distribution: {'GBX_BT': 16, 'GBX_MT': 16, 'GBX_RC': 16, 'GBX_WR': 16, 'NOR': 16}.
- Condition distribution: {'sensor1_spd20hz': 5, 'sensor1_spd25hz': 5, 'sensor1_spd30hz': 5, 'sensor1_spd35hz': 5, 'sensor1_spd40hz': 5, 'sensor1_spd45hz': 5, 'sensor1_spd50hz': 5, 'sensor1_spd55hz': 5, 'sensor2_spd20hz': 5, 'sensor2_spd25hz': 5, 'sensor2_spd30hz': 5, 'sensor2_spd35hz': 5, 'sensor2_spd40hz': 5, 'sensor2_spd45hz': 5, 'sensor2_spd50hz': 5, 'sensor2_spd55hz': 5}.
- Channel_1 RMS is highest on average for 'GBX_WR' and lowest for 'GBX_RC'.
- Channel_1 kurtosis is highest on average for 'GBX_MT', indicating the strongest impulsive behavior.
- Mean inter-channel correlation is -0.031771; mean absolute correlation is 0.059548.
- The strongest outlier candidate is M1_20.MAT, based on IQR feature flags.