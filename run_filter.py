import numpy as np
import matplotlib.pyplot as plt

from utils import (
    GenExp,
    GaussianProcess,
    TopBaseProcessPair,
    ZonalField,
    ParticleFilter,
)

domain = {"xmin": 0, "xmax": 1, "n": 40}

# Define the covariance function
cov_func_base = GenExp(
    {"standard_deviation": 0.15, "correlation_range": 0.30, "exponent": 1.9}
)

cov_func_thickness = GenExp(
    {"standard_deviation": 0.10, "correlation_range": 0.40, "exponent": 1.8}
)


# Define the mean functions
def mean_func_base(x):
    a = 0.67
    b = 0.5
    return a + b * x


def mean_func_thickness(x):
    return np.full_like(x, 0.5)


# Create process pair
gp_base = GaussianProcess(cov_func_base, mean_func_base, domain)
gp_thickness = GaussianProcess(cov_func_thickness, mean_func_thickness, domain)
tbpp = TopBaseProcessPair(gp_base, gp_thickness)

# Create GR field
gr_field = ZonalField([0.0, 0.4, 0.6, 1.0], [0.9, 0.33, 0.67, 0.1])

# Prep well data
well_data = {
    "x": [0.15, 0.25, 0.35],
    "z": [0.5, 0.4, 0.5],
    "gr": [0.5, 0.5, 0.5],
}

# Set up PF
pf = ParticleFilter(
    n_particles=250,
    tbpp=tbpp,
    zf=gr_field,
    noise_params={"sd_gr": 0.10},
    well_data=well_data,
    domain=domain,
)

pf.compute_weights(obs_index=0)
pf.resample()
pf.forecast(obs_index=1)
pf.compute_weights(obs_index=1)
pf.resample()
pf.forecast(obs_index=2)
pf.compute_weights(obs_index=2)
pf.resample()

# show the initial state
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
pf.plot_particles(ax=ax)
plt.show()
