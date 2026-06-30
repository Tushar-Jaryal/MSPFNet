# Dataset 1 EDA Observations

- Dataset 1 contains 60 files with 2 channels per file: ['Channel_1', 'Channel_2'].
- All signals have a consistent length of 2,000,000 samples.
- The file naming pattern successfully encodes fault, condition, and trial information.
- Fault distribution: {'B': 12, 'C': 12, 'H': 12, 'I': 12, 'O': 12}.
- Condition distribution: {'A': 15, 'B': 15, 'C': 15, 'D': 15}.
- Channel 1 RMS is highest on average for fault class 'I' and lowest for 'H'.
- Channel 1 kurtosis is highest on average for fault class 'C', indicating stronger impulsive behavior in that class.
- The mean Channel 1–Channel 2 correlation is 0.000097, with mean absolute correlation 0.000269, suggesting the channels capture different information.
- Channel 2 has an average kurtosis of -1.987, supporting a switching/reference-like signal interpretation.
- The strongest outlier candidate is C-A-1.mat, based on the number of outlier features.