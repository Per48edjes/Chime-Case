import glob
import os

import pandas as pd

from helpers import DATA_PATH

# Load data
data_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
data_dict = {}
for file in data_files:
    file_name = os.path.splitext(os.path.basename(file))[0]
    df_i = pd.read_csv(file)
    data_dict[file_name] = df_i

# Assign to user-friendly variables
df_perf = data_dict["performance"]
df_acq_segment = data_dict["acquisition_segment_counts"]
df_acq_agg = data_dict["acquisition_agg"]

# Set indices
df_acq_agg.set_index("Variant", inplace=True)
df_acq_segment.set_index("Variant", inplace=True)

# Reorder rows for convenience in NHST
df_acq_agg = df_acq_agg.reindex(index=df_acq_agg.index[::-1])
df_acq_segment = df_acq_segment.reindex(index=df_acq_segment.index[::-1])

# Funnel metrics by variant
df_acq_agg["CTR"] = df_acq_agg["Unique Clicks"] / df_acq_agg["Population"]
df_acq_agg["Click CR"] = df_acq_agg["Conversions"] / df_acq_agg["Unique Clicks"]
df_acq_agg["User CR"] = df_acq_agg["CTR"] * df_acq_agg["Click CR"]

# Cost-per-Unique-Click by variant
df_acq_agg["CPUC"] = df_acq_agg["Spend"] / df_acq_agg["Unique Clicks"]

# Cost-per-Conversion by variant
df_acq_agg["CAC"] = df_acq_agg["Spend"] / df_acq_agg["Conversions"]
