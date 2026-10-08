import numpy as np
import pandas as pd
from scipy import stats

# The base model for the basic scenario usiing a log normal fit for all vessel types in a compound poission process.

class BasicScenarioModel:

    """
    Base model for the basic scenario using a log normal fit for all vessel types in a compound Poisson process.

    """

    def __init__(self):

        # Base arrival rates for different vessel types
        self.base_lam_cargo = 2.048
        self.base_lam_tanker = 5.235
        self.base_lam_container = 7.863

        self.params = {
            "Container": {"mu": 9.6477, "sigma": 0.3475},
            "Tanker": {"mu": 10.7971, "sigma": 0.2959},
            "Bulk": {"mu": 10.2272, "sigma": 0.3983}
        }

    def simulate_scenario(self, days=365, num_simulation=10000):

        #Number of vessels per year
        N_container = stats.poisson.rvs(mu=self.base_lam_container, size=(num_simulation, days))
        N_tanker = stats.poisson.rvs(mu=self.base_lam_tanker, size=(num_simulation, days))
        N_cargo = stats.poisson.rvs(mu=self.base_lam_cargo, size=(num_simulation, days))

        N_total = N_container + N_tanker + N_cargo


        # Cargo sums
        S_container = np.zeros((num_simulation, days))
        S_tanker = np.zeros((num_simulation, days))    
        S_cargo = np.zeros((num_simulation, days))  

        #Log normal distribution for container and tanker vessels
        dist_container = stats.lognorm(s=self.params["Container"]["sigma"], scale=np.exp(self.params["Container"]["mu"]))
        dist_tanker = stats.lognorm(s=self.params["Tanker"]["sigma"], scale=np.exp(self.params["Tanker"]["mu"]))
        dist_cargo = stats.lognorm(s=self.params["Bulk"]["sigma"], scale=np.exp(self.params["Bulk"]["mu"]))

        #run simulation:
        for i in range(num_simulation):
            for day in range(days):
                # Pass the single integer of ships for that exact day (N[i, day])
                if N_container[i, day] > 0: S_container[i, day] = np.sum(dist_container.rvs(size=N_container[i, day]))
                if N_tanker[i, day] > 0: S_tanker[i, day] = np.sum(dist_tanker.rvs(size=N_tanker[i, day]))
                if N_cargo[i, day] > 0: S_cargo[i, day] = np.sum(dist_cargo.rvs(size=N_cargo[i, day]))

        # Convert to anual sums for vessels and cargo
        annual_N_container = np.sum(N_container, axis=1)
        annual_N_tanker = np.sum(N_tanker, axis=1)
        annual_N_cargo = np.sum(N_cargo, axis=1)
        annual_N_total = np.sum(N_total, axis=1)

        annual_S_container = np.sum(S_container, axis=1)
        annual_S_tanker = np.sum(S_tanker, axis=1)
        annual_S_cargo = np.sum(S_cargo, axis=1)
        annual_S_total = annual_S_container + annual_S_tanker + annual_S_cargo

        # List of the annual arrays to process in the list comprehensions below
        annual_arrays = [
            annual_N_container, annual_N_tanker, annual_N_cargo, annual_N_total, 
            annual_S_container, annual_S_tanker, annual_S_cargo, annual_S_total
        ]

        # Metrics including mean, std, and 95th percentile for the total annual vessels and cargo
        metrics = {
            "Variable": ["N_container", "N_tanker", "N_cargo", "N_total", "S_container", "S_tanker", "S_cargo", "S_total"],
            "Mean": [np.mean(x) for x in annual_arrays],
            "Std": [np.std(x, ddof=1) for x in annual_arrays], 
            "95th Percentile": [np.percentile(x, 95) for x in annual_arrays]
        }
        return pd.DataFrame(metrics)


if __name__ == "__main__":
    model = BasicScenarioModel()
    metrics = model.simulate_scenario(days=365, num_simulation=10000)
    print(metrics.to_string(index=False))
