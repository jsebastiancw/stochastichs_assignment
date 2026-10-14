import pandas as pd
import numpy as np
from scipy import stats

# our custom model for the scenario using a different fit for each vessel type in a compound poission process.
class VesselArrivalModel:
    def __init__(self):
        # BAse arrival rates for different vessel types
        self.base_lam_cargo = 2.048
        self.base_lam_tanker = 5.235
        self.base_lam_container = 7.863

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
                    "Daily_vessels": np.zeros(num_simulation * days),
                    "Daily_cargo": np.zeros(num_simulation * days),
                    "S_total_matrix": np.zeros((num_simulation, days))
                }

        # Arrays to store annual totals per vessel type (same as base.py)
        annual_N_container = np.zeros(num_simulation)
        annual_N_tanker = np.zeros(num_simulation)
        annual_N_cargo = np.zeros(num_simulation)

        annual_S_container = np.zeros(num_simulation)
        annual_S_tanker = np.zeros(num_simulation)
        annual_S_cargo = np.zeros(num_simulation)

        # fitting each cargo type to a different distribution based on the dataset analysis
        dist_container = stats.lognorm(s=self.sig_container, scale=np.exp(self.mu_container))
        dist_tanker = stats.lognorm(s=self.sig_tanker, scale=np.exp(self.mu_tanker))
        dist_cargo = stats.gamma(a=self.alpha_b, scale=1/self.beta_b)

        # Running the monte carlo simulation for the specified number of simulations
        for i in range(num_simulation):
            cargo_container_total, cargo_tanker_total, cargo_cargo_total = 0, 0, 0
            vessels_container, vessels_tanker, vessels_cargo = 0, 0, 0
            for day in range(days):
                n_container = stats.poisson.rvs(mu=self.base_lam_container)
                n_tanker = stats.poisson.rvs(mu=self.base_lam_tanker)
                n_cargo = stats.poisson.rvs(mu=self.base_lam_cargo)

                vessels_container += n_container
                vessels_tanker += n_tanker
                vessels_cargo += n_cargo

                today_container = np.sum(dist_container.rvs(size=n_container)) if n_container > 0 else 0
                today_tanker = np.sum(dist_tanker.rvs(size=n_tanker)) if n_tanker > 0 else 0
                today_cargo = np.sum(dist_cargo.rvs(size=n_cargo)) if n_cargo > 0 else 0

                cargo_container_total += today_container
                cargo_tanker_total += today_tanker
                cargo_cargo_total += today_cargo

                day_index = i * days + day
                results["Daily_vessels"][day_index] = n_container + n_tanker + n_cargo
                results["Daily_cargo"][day_index] = today_container + today_tanker + today_cargo
                results["S_total_matrix"][i, day] = today_container + today_tanker + today_cargo

            #aggregate results for this simulation
            annual_N_container[i] = vessels_container
            annual_N_tanker[i] = vessels_tanker
            annual_N_cargo[i] = vessels_cargo
            results["Total_vessels"][i] = vessels_container + vessels_tanker + vessels_cargo

            annual_S_container[i] = cargo_container_total
            annual_S_tanker[i] = cargo_tanker_total
            annual_S_cargo[i] = cargo_cargo_total
            results["Total_cargo"][i] = cargo_container_total + cargo_tanker_total + cargo_cargo_total

        annual_arrays = [
            annual_N_container, annual_N_tanker, annual_N_cargo, results["Total_vessels"],
            annual_S_container, annual_S_tanker, annual_S_cargo, results["Total_cargo"]
        ]

        metrics = {
            "Variable": ["N_container", "N_tanker", "N_cargo", "N_total", "S_container", "S_tanker", "S_cargo", "S_total"],
            "Mean": [np.mean(x) for x in annual_arrays],
            "95% CI": [1.96 * np.std(x, ddof=1) / np.sqrt(num_simulation) for x in annual_arrays],
            "Std": [np.std(x, ddof=1) for x in annual_arrays],
            "95th Percentile": [np.percentile(x, 95) for x in annual_arrays]
        }
        results["metrics_df"] = pd.DataFrame(metrics)
        
        return results

if __name__ == "__main__":
    model = VesselArrivalModel()
    metrics = model.simulate_scenario(days=365, num_simulation=10000)
    
    print("--- OUR MODEL ANNUAL RESULTS ---")
    print(metrics["metrics_df"].to_string(index=False))

    # Calculate the assignment requirements from the daily arrays
    daily_cargo = metrics["Daily_cargo"]
    daily_vessels = metrics["Daily_vessels"]
    s_matrix = metrics["S_total_matrix"]
    num_runs = s_matrix.shape[0]

    daily_cargo_ci = 1.96 * np.std(np.mean(s_matrix, axis=1), ddof=1) / np.sqrt(num_runs)
    cong_days_650 = np.sum(s_matrix > 650000, axis=1)
    cong_prob = np.mean(cong_days_650 / 365)
    cong_prob_ci = 1.96 * np.std(cong_days_650 / 365, ddof=1) / np.sqrt(num_runs)
    cong_days_800 = np.sum(s_matrix > 800000, axis=1)

    print("\n--- OUR MODEL DAILY & CONGESTION RESULTS ---")
    print(f"Average Daily Vessels: {np.mean(daily_vessels):,.2f} (Std Dev: {np.std(daily_vessels):,.2f})")
    print(f"Average Daily Cargo: {np.mean(daily_cargo):,.2f} ± {daily_cargo_ci:,.2f}")
    print(f"Throughput Variability (Std Dev): {np.std(daily_cargo):,.2f}")
    print(f"95th Percentile Daily Cargo: {np.percentile(daily_cargo, 95):,.2f}")
    print(f"Extreme Outcome (99th Percentile): {np.percentile(daily_cargo, 99):,.2f}")
    print(f"Congestion Probability (>650k tons): {cong_prob:.4%} ± {cong_prob_ci:.4%} ({np.mean(cong_days_650):.2f} days/year)")
    print(f"Congestion Probability (>800k tons): {np.mean(cong_days_800 / 365):.4%} ({np.mean(cong_days_800):.2f} days/year)")