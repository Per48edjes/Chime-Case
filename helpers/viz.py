from matplotlib.ticker import FuncFormatter


@FuncFormatter
def bps(x, pos):
    return f"{x * 10_000:.0f} bps"
