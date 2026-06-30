# D4 EDA Observations

- D4 contains 240 files, 8 loaded channels per file on average, and 8 fault classes.
- Primary diagnostic channel selected by feature-separation score: Channel_6.
- Fault distribution: {'GBX_BT': 36, 'GBX_CK': 36, 'GBX_MT': 12, 'GBX_PT': 36, 'GBX_WR': 36, 'MIX_TI': 36, 'MIX_TO': 36, 'NOR': 12}.
- Condition distribution: {'modespeed_load10Nm_1000rpm': 20, 'modespeed_load10Nm_2000rpm': 20, 'modespeed_load10Nm_3000rpm': 20, 'modespeed_load20Nm_1000rpm': 20, 'modespeed_load20Nm_2000rpm': 20, 'modespeed_load20Nm_3000rpm': 20, 'modetorque_load10Nm_1000rpm': 20, 'modetorque_load10Nm_2000rpm': 20, 'modetorque_load10Nm_3000rpm': 20, 'modetorque_load20Nm_1000rpm': 20, 'modetorque_load20Nm_2000rpm': 20, 'modetorque_load20Nm_3000rpm': 20}.
- Channel_6 RMS is highest on average for 'GBX_MT' and lowest for 'MIX_TO'.
- Channel_6 kurtosis is highest on average for 'GBX_MT', indicating the strongest impulsive behavior.
- Mean inter-channel correlation is 0.018705; mean absolute correlation is 0.103415.
- The strongest outlier candidate is miss_teeth_speed_circulation_20Nm-1000rpm.csv, based on IQR feature flags.