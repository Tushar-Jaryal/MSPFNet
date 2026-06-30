# D5 EDA Observations

- D5 contains 64 files, 3 loaded channels per file on average, and 15 fault classes.
- Primary diagnostic channel selected by feature-separation score: Channel_2.
- Fault distribution: {'BRG_BF': 4, 'BRG_CF': 4, 'BRG_IO': 4, 'BRG_IR': 4, 'BRG_OR': 4, 'GBX_BT': 4, 'GBX_EG': 4, 'GBX_WR': 4, 'MIX_IB': 4, 'MIX_II': 4, 'MIX_IS': 4, 'MIX_OB': 4, 'MIX_OI': 4, 'MIX_OS': 4, 'NOR': 8}.
- Condition distribution: {'1200rpm': 16, '1800rpm': 16, '200-2400-200': 16, '2400rpm': 16}.
- Channel_2 RMS is highest on average for 'BRG_BF' and lowest for 'MIX_IS'.
- Channel_2 kurtosis is highest on average for 'BRG_IR', indicating the strongest impulsive behavior.
- Mean inter-channel correlation is -0.015609; mean absolute correlation is 0.083795.
- The strongest outlier candidate is IF_2400.csv, based on IQR feature flags.