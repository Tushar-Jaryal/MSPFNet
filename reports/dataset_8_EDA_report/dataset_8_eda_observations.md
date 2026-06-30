# D8 EDA Observations

- D8 contains 90 files, 4 loaded channels per file on average, and 3 fault classes.
- Primary diagnostic channel selected by feature-separation score: Channel_2.
- Fault distribution: {'GBX_BT': 30, 'GBX_MT': 30, 'NOR': 30}.
- Condition distribution: {'L0_VS_0_40': 3, 'L1_VS_0_40': 3, 'L2_VS_0_40': 3, 'L3_VS_0_40': 3, 'L4_VS_0_40': 3, 'spd20hz': 15, 'spd25hz': 15, 'spd30hz': 15, 'spd35hz': 15, 'spd40hz': 15}.
- Channel_2 RMS is highest on average for 'GBX_MT' and lowest for 'GBX_BT'.
- Channel_2 kurtosis is highest on average for 'GBX_MT', indicating the strongest impulsive behavior.
- Mean inter-channel correlation is -0.092789; mean absolute correlation is 0.189418.
- The strongest outlier candidate is M_35_3.txt, based on IQR feature flags.