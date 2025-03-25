import numpy as np
import matplotlib.pyplot as plt

from utils import CovarianceFunction, GenExp, GaussianProcess, TopBaseProcessPair

domain = {"xmin": 0, "xmax": 1, "n": 100}

# Define the covariance function
cov_func_base = GenExp(
    {"standard_deviation": 0.15, "correlation_range": 0.33, "exponent": 1.9}
)

cov_func_thickness = GenExp(
    {"standard_deviation": 0.10, "correlation_range": 0.44, "exponent": 1.8}
)


# Define the mean functions
def mean_func_base(x):
    a = 0.67
    b = 0.5
    return a + b * x


def mean_func_thickness(x):
    return np.full_like(x, 0.5)


# Create a Gaussian processes
gp_base = GaussianProcess(cov_func_base, mean_func_base, domain)
gp_thickness = GaussianProcess(cov_func_thickness, mean_func_thickness, domain)
tbpp = TopBaseProcessPair(gp_base, gp_thickness)

# Draw some samples from the Gaussian process
samples = [tbpp.draw_full() for _ in range(1)]

# Plot the samples
plt.figure()

for sample in samples:
    plt.plot(tbpp.xvals, sample["top"], "g-", alpha=0.5)
    plt.plot(tbpp.xvals, sample["base"], "r-", alpha=0.5)


plt.plot(tbpp.xvals, tbpp.mean_top(), "g-", linewidth=1)
plt.plot(tbpp.xvals, tbpp.mean_base(), "r-", linewidth=1)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Sample from Gaussian Process")

plt.ylim(tbpp.limits(n_std=4))

plt.gca().invert_yaxis()
plt.show()
