import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(_current_dir, "..", "data"))


# Revenue
INTERCHANGE = 1.5 / 100  # per purchase dollar

# Costs
COGS = 1.05 / 100  # per purchase dollar
SPEND_TRACKER_PMPM = 5  # cost per member-month

# Profitability
GROSS_MARGIN = INTERCHANGE - COGS
