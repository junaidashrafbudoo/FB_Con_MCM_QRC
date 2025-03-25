import numpy as np
import matplotlib.pyplot as plt

alpha, beta, gamma, td = 0.2, 10, 0.1, 17


class MackeyGlassSequence:
    def __init__(self, alpha, beta, gamma, td):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.td = td

    def get_mackey_glass_sequence(self, N=2000, x0=1.2):
        # Convert the continuous delay t_d to an integer delay tau
        # assuming we are stepping by Δt = 1.
        tau = int(self.td)

        # We need a buffer to handle x[n - tau] for the initial steps.
        # We'll allocate N+tau steps and fill the first 'tau' values
        # with some initial condition (constant or random).
        x = np.zeros(N + tau)
        x[:tau] = x0  # simple constant initialization

        # Generate the sequence using the forward-Euler approximation
        for n in range(tau, N + tau):
            # x[n-1] corresponds to x(n) in eqn
            # x[n]   corresponds to x(n+1)
            x[n] = x[n - 1] + self.alpha * x[n - tau] / (1.0 + x[n - tau] ** self.beta) - self.gamma * x[n - 1]

        mg_series = x[tau:]
        # Return only the last N points (discard the first tau "warm-up" points)
        # 1. Compute the min and max
        min_val = np.min(mg_series)
        max_val = np.max(mg_series)

        # 2. Scale the entire sequence into [0, 1]
        mg_series_norm = (mg_series - min_val) / (max_val - min_val)
        return mg_series_norm

"""
seq = MackeyGlassSequence(0.2, 10, 0.1, 17)
l = seq.get_mackey_glass_sequence(N=3710)
plt.plot(l[3000:])
plt.show()"""
