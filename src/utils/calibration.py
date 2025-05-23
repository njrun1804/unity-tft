import numpy as np
import matplotlib.pyplot as plt

def pit_values(y_true: np.ndarray, samples: np.ndarray):
    return (samples < y_true).mean(axis=0)   # one PIT per horizon

def pit_histogram(ax, pit_vals, bins=10):
    ax.hist(pit_vals, bins=bins, density=True, edgecolor="black")
    ax.axhline(1, ls="--")
    ax.set(title="PIT Histogram", xlabel="PIT", ylabel="Density")
