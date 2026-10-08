import pandas as pd
import numpy as np
from scipy import stats

# our custom model for the scenario using a different fit for each vessel type in a compound poission process.

class VesselArrivalModel:

    def __init__(self):

        # BAse arrival rates for different vessel types
        self.base_lam_cargo = 7.863
        self.base_lam_tanker = 5.235
        self.base_lam_container = 2.048

        # seasonality multipliers based on day of the week using the dataset analysis
        inter_arrival_hours = [1.567, 1.557, 1.580, 1.635, 1.578, 1.570, 1.606]
        self.daily_multipliers = [(24 / hours) / 15.14 for hours in inter_arrival_hours]

        # Cargro volume parametets for container and tanker vessels using the log normal fit from the dataset analysis
        self.mu_container, self.sig_container = 9.6477, 0.3475
        self.mu_tanker, self.sig_tanker = 10.7971, 0.2959

        # Cargo volume parameters for cargo vessles using the gamma fit from the dataset analysis
        m1_bulk = 29926.481
        std_bulk = 12409.654
        var_bulk = std_bulk ** 2
        self.alpha_b = (m1_bulk ** 2) / var_bulk
        self.beta_b = m1_bulk / var_bulk


    def simulate_scenario(self, days = 365, num_simulation = 10000):
        """
        Simulate the vessel arrivals and cargo volumes for a given number of days and simulations.
        
        """

        results = {

            "Total_vessels": np.zeros(num_simulation),
            "Total_cargo": np.zeros(num_simulation),

        }

        # fitting each cargo type to a different distribution based on the dataset analysis

        dist_container = stats.lognorm(s=self.sig_container, scale=np.exp(self.mu_container))
        dist_tanker = stats.lognorm(s=self.sig_tanker, scale=np.exp(self.mu_tanker))
        dist_cargo = stats.gamma(a=self.alpha_b, scale=1/self.beta_b)

        # Running the monte carlo simulation for the specified number of simulations

        for i in range(num_simulation):
            cargo_container_total, cargo_tanker_total, cargo_cargo_total = 0, 0, 0
            vessels_container, vessels_tanker, vessels_cargo = 0, 0, 0

            for day in range(days):
                #Determine day of week
                day_of_week = day % 7

                multiplier = self.daily_multipliers[day_of_week]

                n_container = stats.poisson.rvs(mu=self.base_lam_container * multiplier)
                n_tanker = stats.poisson.rvs(mu=self.base_lam_tanker * multiplier)
                n_cargo = stats.poisson.rvs(mu=self.base_lam_cargo * multiplier)

                vessels_container += n_container
                vessels_tanker += n_tanker
                vessels_cargo += n_cargo

                # draw cargo volumes for each vessel type based on the fitted distributions
                if n_container > 0: cargo_container_total += np.sum(dist_container.rvs(size=n_container))
                if n_tanker > 0: cargo_tanker_total += np.sum(dist_tanker.rvs(size=n_tanker))
                if n_cargo > 0: cargo_cargo_total += np.sum(dist_cargo.rvs(size=n_cargo))


            #aggregate results for this simulation
            results["Total_vessels"][i] = vessels_container + vessels_tanker + vessels_cargo
            results["Total_cargo"][i] = cargo_container_total + cargo_tanker_total + cargo_cargo_total

        return results


if __name__ == "__main__":
    model = VesselArrivalModel()
    sim_data = model.simulate_scenario(days=365, num_simulation=10000)
    print(f"Average total vessels over 365 days: {np.mean(sim_data['Total_vessels']):.2f}")

    
    

